// This program converts a set of images and annotations to a lmdb/leveldb by
// storing them as AnnotatedDatum proto buffers.
// Usage:
//   convert_annoset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images and
// annotations, and LISTFILE should be a list of files as well as their labels
// or label files.
// For classification task, the file should be in the format as
//   imgfolder1/img1.JPEG 7
//   ....
// For detection task, the file should be in the format as
//   imgfolder1/img1.JPEG annofolder1/anno1.xml
//   ....

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <map>
#include <string>
#include <utility>//utility头文件定义了一个pair类型,pair类型用于存储一对数据
#include <vector>

#include "boost/scoped_ptr.hpp"//智能指针头文件//忘记手动释放内存了，也没事儿，不会造成内存泄漏.在这个时候，智能指针的出现实际上就是为了可以方便的控制对象的生命期，在智能指针中，一个对象什么时候和在什么条件下要被析构或者是删除是受智能指针本身决定的，用户并不需要管理。
#include "boost/variant.hpp"//variant is "multi-type, single value"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"//引入包装好的lmdb操作函数
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"//引入opencv中的图像操作函数,ReadImageToDatum函数为io.cpp文件中定义的函数；io.cpp主要实现了3部分功能：1，从text文件或者二进制文件中读写proto文件；2，利用opencv的Mat矩阵，把图像数据读到Mat矩阵中；3，把Mat矩阵中的值放入到datum中
#include "caffe/util/rng.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

DEFINE_bool(gray, false,
    "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "lmdb",
    "The backend {lmdb, leveldb} for storing the result");
DEFINE_string(anno_type, "classification",
    "The type of annotation {classification, detection}.");
DEFINE_string(label_type, "xml",
    "The type of annotation file format.");
DEFINE_string(label_map_file, "",
    "A file with LabelMap protobuf message.");
DEFINE_bool(check_label, false,
    "When this option is on, check that there is no duplicated name/label.");
DEFINE_int32(min_dim, 0,
    "Minimum dimension images are resized to (keep same aspect ratio)");
DEFINE_int32(max_dim, 0,
    "Maximum dimension images are resized to (keep same aspect ratio)");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");
DEFINE_bool(check_size, false,
    "When this option is on, check that all the datum have the same size");
DEFINE_bool(encoded, false,
    "When this option is on, the encoded image will be save in datum");
DEFINE_string(encode_type, "",
    "Optional: What type should we encode the image as ('png','jpg',...).");

int main(int argc, char** argv) {
#ifdef USE_OPENCV
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of images and annotations to the "
        "leveldb/lmdb format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_annoset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_annoset");
    return 1;
  }

  const bool is_color = !FLAGS_gray;
  const bool check_size = FLAGS_check_size;
  const bool encoded = FLAGS_encoded;
  const string encode_type = FLAGS_encode_type;
  const string anno_type = FLAGS_anno_type;
  AnnotatedDatum_AnnotationType type;
  const string label_type = FLAGS_label_type;
  const string label_map_file = FLAGS_label_map_file;
  const bool check_label = FLAGS_check_label;
  std::map<std::string, int> name_to_label;

  std::ifstream infile(argv[2]);//创建指向train.txt文件的文件读入流
  std::vector<std::pair<std::string, boost::variant<int, std::string> > > lines;//pair 是 一种模版类型。每个pair 可以存储两个值。这两种值无限制。也可以将自己写的struct的对象放进去。
  std::string filename;
  int label;
  std::string labelname;
  if (anno_type == "classification") {
    while (infile >> filename >> label) {//infile>>name 是在basic_istream类里面定义的一个函数 oprator>>（……）函数. infile >> filename >> label 和infile >> filename;infile >> label;是一样的它是以换行做为读取一行结束的标记
      lines.push_back(std::make_pair(filename, label));
    }
  } else if (anno_type == "detection") {
    type = AnnotatedDatum_AnnotationType_BBOX;
    LabelMap label_map;
    CHECK(ReadProtoFromTextFile(label_map_file, &label_map))
        << "Failed to read label map file.";
    CHECK(MapNameToLabel(label_map, check_label, &name_to_label))
        << "Failed to convert name to label.";
    while (infile >> filename >> labelname) {
      lines.push_back(std::make_pair(filename, labelname));
    }
  }
  if (FLAGS_shuffle) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    shuffle(lines.begin(), lines.end());
  }
  LOG(INFO) << "A total of " << lines.size() << " images.";

  if (encode_type.size() && !encoded)
    LOG(INFO) << "encode_type specified, assuming encoded=true.";

  int min_dim = std::max<int>(0, FLAGS_min_dim);
  int max_dim = std::max<int>(0, FLAGS_max_dim);
  int resize_height = std::max<int>(0, FLAGS_resize_height);
  int resize_width = std::max<int>(0, FLAGS_resize_width);

  // Create new DB
  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));//智能指针的创建方式类似泛型的格式，上面通过db.cpp内定义的命名的子命名空间中db的“成员函数”GetDB函数来初始化db对象
  db->Open(argv[3], db::NEW);//argv[3]的文件夹下创建并打开lmdb的操作环境
  scoped_ptr<db::Transaction> txn(db->NewTransaction());//创建lmdb文件的操作句柄

  // Storing to db
  std::string root_folder(argv[1]);
  AnnotatedDatum anno_datum;//see https://zhuanlan.zhihu.com/p/22404295
  Datum* datum = anno_datum.mutable_datum();
  int count = 0;
  int data_size = 0;
  bool data_size_initialized = false;

  for (int line_id = 0; line_id < lines.size(); ++line_id) {
    bool status = true;
    std::string enc = encode_type;
    if (encoded && !enc.size()) {
      // Guess the encoding type from the file name
      string fn = lines[line_id].first;
      size_t p = fn.rfind('.');//反向查找，'.'在fn中最后出现的位置
      if ( p == fn.npos )//npos 是一个常数，用来表示不存在的位置，类型一般是std::container_type::size_type 许多容器都提供这个东西。取值由实现决定，一般是-1，这样做，就不会存在移植的问题了。
        LOG(WARNING) << "Failed to guess the encoding of '" << fn << "'";
      enc = fn.substr(p);//s.substr(5); //只有一个数字5表示从下标为5开始一直到结尾
      std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);//string转为小写//begin:指向字符串的第一个元素//array<int,8> t1 = {3,5,7,11,13,17,19,23};array<int,8> t2; //将t1中所有元素加1，并赋给t2  transform(t1.begin(),t1.end(),t2.begin(),[](int i){return ++i;});//此时t2={4,6,8,12,14,18,20,24},t1不变
    }
    filename = root_folder + lines[line_id].first;//full path to image
    if (anno_type == "classification") {
      label = boost::get<int>(lines[line_id].second);//boost::get<int> 用来读取boost::variant
      status = ReadImageToDatum(filename, label, resize_height, resize_width,////把图像数据读取到datum中
          min_dim, max_dim, is_color, enc, datum);
    } else if (anno_type == "detection") {
      labelname = root_folder + boost::get<std::string>(lines[line_id].second);
      status = ReadRichImageToAnnotatedDatum(filename, labelname, resize_height,//see io.cpp
          resize_width, min_dim, max_dim, is_color, enc, type, label_type,
          name_to_label, &anno_datum);
      anno_datum.set_type(AnnotatedDatum_AnnotationType_BBOX);
    }
    if (status == false) {
      LOG(WARNING) << "Failed to read " << lines[line_id].first;
      continue;
    }
    if (check_size) {//检查图片尺寸
      if (!data_size_initialized) {{//若data_size_initialized没有初始化
        data_size = datum->channels() * datum->height() * datum->width();
        data_size_initialized = true;
      } else {
        const std::string& data = datum->data();
        CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
            << data.size();
      }
    }
    // sequential
    string key_str = caffe::format_int(line_id, 8) + "_" + lines[line_id].first;

    // Put in db
    string out;
    CHECK(anno_datum.SerializeToString(&out));
    txn->Put(key_str, out);

    if (++count % 1000 == 0) {
      // Commit db
      txn->Commit();
      txn.reset(db->NewTransaction());
      LOG(INFO) << "Processed " << count << " files.";
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    txn->Commit();
    LOG(INFO) << "Processed " << count << " files.";
  }
#else
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  return 0;
}
