#include "kernel_iou.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

// CPU specialization of actual computation.
template <typename T>
struct IOUFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, int size, const T* in, T* out) {
    for (int i = 0; i < size; ++i) {
      out[i] = 2 * in[i];
    }
  }
};

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class IOUOp : public OpKernel {
public:
  explicit IOUOp(OpKernelConstruction* context) : OpKernel(context) {}

void Compute(OpKernelContext* context) override {
    // some checks to be sure ...
    DCHECK_EQ(2, context->num_inputs());

    // get the input tensor
    const Tensor& box_a_tensor = context->input(0);
    const Tensor& box_b_tensor = context->input(1);

    OP_REQUIRES(context, TensorShapeUtils::IsVector(box_a_tensor.shape()),
                errors::InvalidArgument("iou expects a 1-D vector for box_a."));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(box_b_tensor.shape()),
                errors::InvalidArgument("iou expects a 1-D vector for box_b."));

    // create output
    TensorShape output_shape;
    output_shape.Clear();
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));

    IOUFunctor<Device, T>()(
        context->eigen_device<Device>(),
        static_cast<int>(input_tensor.NumElements()),
        input_tensor.flat<T>().data(),
        output_tensor->flat<T>().data());
  }
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("iou").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      IOUOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(int32);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  /* Declare explicit instantiations in kernel_iou.cu.cc. */ \
  extern template IOUFunctor<GPUDevice, T>;                  \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("iou").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      IOUOp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(int32);
#endif  // GOOGLE_CUDA

