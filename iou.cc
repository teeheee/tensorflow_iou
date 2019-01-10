#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <iostream>
#include <math.h>
#include <vector>

using namespace std;

using namespace tensorflow;

REGISTER_OP("IOU")
  .Input("box_a: float")
  .Input("box_b: float")
  .Output("iou_out: float")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    c->set_output(0, c->Scalar());
    return Status::OK();
  });

class Vector{
public:
    double x,y;
    Vector(){
      y=x=0;
    };
    Vector(const Vector& v){
      x=v.x;
      y=v.y;
    };
    Vector(double ax, double ay){
      x = ax;
      y = ay;
    };
    Vector operator+(Vector& v){
      return Vector(x + v.x, y + v.y);
    };
    Vector operator-(Vector& v){
      return Vector(x + v.x, y + v.y);
    };
    double cross(Vector& v){
      return x*v.y - y*v.x;
    };
    void print(){
      cout << "[" << x << ", " << y   <<"],"  << endl;
    };
};

class Line{
public:
  // ax + by + c = 0
  double a,b,c;
  Line(Vector &v1, Vector &v2){
        a = v2.y - v1.y;
        b = v1.x - v2.x;
        c = v2.cross(v1);
  };
  double value(Vector p) {
        return a*p.x + b*p.y + c;
  }
  Vector intersection(Line& other){
      double w = a*other.b - b*other.a;
      double x = (b*other.c - c*other.b)/w;
      double y = (c*other.a - a*other.c)/w;
      return Vector(x, y);
  };
  void print(){
    cout << "[" << a << ", " << b << ", "  << c << ", "  <<"],"  << endl;
  };
};


class IOUOp : public OpKernel {
public:
  explicit IOUOp(OpKernelConstruction* context) : OpKernel(context) {}

  vector<Vector> rectangle_vertices(double cx, double cy, double w, double h, double r){
      double dx = w/2;
      double dy = h/2;
      double dxcos = dx*cos(r);
      double dxsin = dx*sin(r);
      double dycos = dy*cos(r);
      double dysin = dy*sin(r);
      Vector a(-dxcos - -dysin + cx, -dxsin + -dycos + cy);
      Vector b( dxcos - -dysin + cx,  dxsin + -dycos + cy);
      Vector c( dxcos -  dysin + cx,  dxsin +  dycos + cy);
      Vector d(-dxcos -  dysin + cx, -dxsin +  dycos + cy);
      vector<Vector> return_type {a,b,c,d};
      return return_type;
  };

  void print(vector<Vector> vector_list, const char* name){
    cout << name << "_cpp =  np.array([" << endl;
    for(int i = 0; i < vector_list.size(); i++){
      vector_list[i].print();
    }
    cout << "])" << endl;
  };

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
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_shape, &output_tensor));

    // get the corresponding Eigen tensors for data access
    auto box_a = box_a_tensor.vec<float>();
    auto box_b = box_b_tensor.vec<float>();
    auto output = output_tensor->scalar<float>();

    vector<Vector> rect_1 = rectangle_vertices(
      box_a(0),box_a(1),box_a(2),box_a(3),box_a(4));
    vector<Vector> rect_2 = rectangle_vertices(
      box_b(0),box_b(1),box_b(2),box_b(3),box_b(4));

    vector<Vector> intersection = rect_1;

    int rec_2_length = rect_2.size();
    for(int rec_2_index = 0; rec_2_index < rec_2_length; rec_2_index++){

      //  print(intersection,"intersection");
        if(intersection.size() <= 2)
          break; //no intersection
        Vector p = rect_2[rec_2_index];
        Vector q = rect_2[(rec_2_index+1)%rec_2_length];

        Line line = Line(p, q);
        vector<Vector> new_intersection;
        int inter_length = intersection.size();
        vector<double> line_values;
        for(int inter_index = 0; inter_index < inter_length; inter_index++){
            line_values.push_back(line.value(intersection[inter_index]));
        }

        for(int inter_index = 0; inter_index < inter_length; inter_index++)
        {
            Vector s = intersection[inter_index];
            Vector t = intersection[(inter_index+1)%inter_length];
            double s_value = line_values[inter_index];
            double t_value = line_values[(inter_index+1)%inter_length];
            if(s_value <= 0){
              new_intersection.push_back(s);
            }
            if(s_value * t_value < 0){
              Line line_2(s, t);
              new_intersection.push_back(line.intersection(line_2));
            }
        }
        intersection = new_intersection;
    }

    double intersec_val = 0;
    if(intersection.size() > 2){
      int inter_length = intersection.size();
      double sum = 0;
      for(int inter_index = 0; inter_index < inter_length; inter_index++)
      {
        Vector s = intersection[inter_index];
        Vector t = intersection[inter_index] + intersection[(inter_index+1)%inter_length];
        sum += s.cross(t);
      }
      intersec_val = 0.5*sum;
    }

    double union_val = box_b(2)*box_b(3) + box_a(2)*box_a(3);

    output(0) = intersec_val/(union_val-intersec_val);
   };
};

REGISTER_KERNEL_BUILDER(Name("IOU").Device(DEVICE_CPU), IOUOp);
