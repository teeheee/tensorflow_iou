#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("IOU")
  .Input("box_a: float")
  .Input("box_b: float")
  .Output("iou_out: float")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    c->set_output(0, c->Scalar());
    return Status::OK();
  });


class IOUOp : public OpKernel {
public:
  explicit IOUOp(OpKernelConstruction* context) : OpKernel(context) {}
  
  void Compute(OpKernelContext* context) override {
    // some checks to be sure ...
    DCHECK_EQ(2, context->num_inputs());
    
    // get the input tensor
    const Tensor& box_a = context->input(0);
    const Tensor& box_b = context->input(1);
    
    // check shapes of input and weights
    const TensorShape& a_shape = box_a.shape();
    const TensorShape& b_shape = box_b.shape();
    DCHECK_EQ(a_shape.dims(), 1);
    DCHECK_EQ(a_shape.dim_size(0), 7);
    DCHECK_EQ(b_shape.dims(), 1);
    DCHECK_EQ(b_shape.dim_size(0), 7);
    
    // create output shape
    TensorShape output_shape;
    //output_shape.AddDim(1);
            
    // create output tensor
    Tensor* output = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    
    // get the corresponding Eigen tensors for data access
    auto box_a_tensor = box_a.matrix<float>();
    auto box_b_tensor = box_b.matrix<float>();
    auto output_tensor = output->matrix<float>();
    
    //TODO iou calculation
    output_tensor(0)=1;
  }
};

REGISTER_KERNEL_BUILDER(Name("IOU").Device(DEVICE_CPU), IOUOp);
