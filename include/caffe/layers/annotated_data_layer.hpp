#ifndef CAFFE_DATA_LAYER_HPP_
#define CAFFE_DATA_LAYER_HPP_

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

namespace caffe {

template <typename Dtype>//模板类
class AnnotatedDataLayer : public BasePrefetchingDataLayer<Dtype> {//继承
 public:
  explicit AnnotatedDataLayer(const LayerParameter& param);//explicit构造函数必须显式调用
  virtual ~AnnotatedDataLayer();//虚函数是指一个类中你希望重载的成员函数，当你用一个基类指针或引用指向一个继承类对象的时候，你调用一个虚函数，实际调用的是继承类的版本。
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,//数据输入层的参数设置
      const vector<Blob<Dtype>*>& top);
  // AnnotatedDataLayer uses DataReader instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return false; }//inline c++ inline 用函数体代替调用处的函数名，用在多次调用的小函数上，防止消耗大量栈空间
  virtual inline const char* type() const { return "AnnotatedData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }//因为已经是最底层了
  virtual inline int MinTopBlobs() const { return 1; }//至少有img

 protected://基类对象不能访问基类的protected成员，派生类中可以访问基类的protected成员
  virtual void load_batch(Batch<Dtype>* batch);

  DataReader<AnnotatedDatum> reader_;
  bool has_anno_type_;
  AnnotatedDatum_AnnotationType anno_type_;
  vector<BatchSampler> batch_samplers_;
  string label_map_file_;
};

}  // namespace caffe

#endif  // CAFFE_DATA_LAYER_HPP_
