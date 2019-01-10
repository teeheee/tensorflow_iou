#!/usr/bin/env python3

import unittest
import numpy as np
import tensorflow as tf
import iou

iou_module = tf.load_op_library('./libiou.so')
iou_tf_function = iou_module.iou

scale_vec = np.array([40,80,2,2,2*3.1415])

class IOU_test(unittest.TestCase):
   '''
   def test_raisesExceptionWithIncompatibleDimensions(self):
        with tf.Session(''):
            with self.assertRaises(ValueError):
                iou_tf_function([1,1],[1,2,3]).eval()
            with self.assertRaises(ValueError):
                iou_tf_function([[1,1],[1,1]],[1,2,3]).eval()
   '''
   '''
   def test_problem_case(self):
       box_a = np.array([3.73375588, 0.79769908, 1.7761505,  3.42481639, 3.69286412])
       box_b = np.array([2.7952896,  2.46695814, 1.99869914, 2.54956993, 5.69758454])
       with tf.Session('') as sess:
             box_a_p = tf.placeholder(tf.float32, shape = (5))
             box_b_p = tf.placeholder(tf.float32, shape = (5))

             iou_tf = iou_tf_function(box_a_p,box_b_p)

             for i in range(100000):
                 print("-------------------------")

                 iou_module = sess.run(iou_tf, feed_dict = {box_a_p: box_a, box_b_p: box_b})
                 iou_python = iou.iou_pure_python(box_a,box_b)

                 print("iou_module = %f"%iou_module)
                 print("iou_python = %f"%iou_python)

                 np.testing.assert_almost_equal(iou_python, iou_module, decimal=2)
   '''
   def test_speed(self):
       import time

       with tf.Session('') as sess:
           box_a_p = tf.placeholder(tf.float32, shape = (5))
           box_b_p = tf.placeholder(tf.float32, shape = (5))
           iou_tf = iou_tf_function(box_a_p,box_b_p)

           time_start = time.clock()
           for i in range(100):
               a = np.random.rand(5)*scale_vec
               b = np.random.rand(5)*scale_vec
               iou_module = sess.run(iou_tf, feed_dict = {box_a_p: a, box_b_p: b})
           time_elapsed = (time.clock() - time_start)
           print("tensorflow version took %fs"%time_elapsed)

           time_start = time.clock()
           for i in range(100):
               a = np.random.rand(5)*scale_vec
               b = np.random.rand(5)*scale_vec
               iou_python = iou.iou_pure_python(a,b)
           time_elapsed = (time.clock() - time_start)
           print("python version took %fs"%time_elapsed)

   def test_random_box(self):
      with tf.Session('') as sess:
            box_a_p = tf.placeholder(tf.float32, shape = (5))
            box_b_p = tf.placeholder(tf.float32, shape = (5))



            iou_tf = iou_tf_function(box_a_p,box_b_p)

            for i in range(100):
                a = np.random.rand(5)*scale_vec
                b = np.random.rand(5)*scale_vec

                iou_module = sess.run(iou_tf, feed_dict = {box_a_p: a, box_b_p: b})
                iou_python = iou.iou_pure_python(a,b)


                np.testing.assert_almost_equal(iou_python, iou_module, decimal=2)


if __name__ == '__main__':
    unittest.main()
