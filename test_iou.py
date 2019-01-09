#!/usr/bin/env python3

import unittest
import numpy as np
import tensorflow as tf
import iou

iou_module = tf.load_op_library('/home/ulbrich/test-workspace/iou_tensorflow_op/libiou.so')
iou_tf_function = iou_module.iou

class IOU_test(unittest.TestCase):
   '''   
   def test_raisesExceptionWithIncompatibleDimensions(self):
        with tf.Session(''):
            with self.assertRaises(ValueError):
                iou_tf_function([1,1],[1,2,3]).eval()
            with self.assertRaises(ValueError):
                iou_tf_function([[1,1],[1,1]],[1,2,3]).eval()
   '''
            
   def test_random_box(self):
      with tf.Session('') as sess:
            box_a_p = tf.placeholder(tf.float32, shape = (7))
            box_b_p = tf.placeholder(tf.float32, shape = (7))
            
            

            iou = iou_tf_function(box_a_p,box_b_p)
            
            for i in range(100):
                a = np.random.randint(10, size = (7))
                b = np.random.randint(10, size = (7))
                
                iou_module = sess.run(iou, feed_dict = {box_a_p: a, box_b_p: b})
                iou_python = iou_pure_python(a,b)
                
                np.testing.assert_array_equal(iou_python, iou_module)
                  
                
if __name__ == '__main__':
    unittest.main()
