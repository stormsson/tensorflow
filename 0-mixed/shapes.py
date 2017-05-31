import tensorflow as tf
import numpy as np

def example():
    input_pl = tf.placeholder(tf.float32, [None, None, 3])
    basic_a = [ [ .1,.2,.3], [.4,.5,.6], [.7,.8,.9]]
    basic_b = [ [ .5,.5,.5] ]
    a = np.ndarray((3, 3,), buffer=np.array(basic_a, dtype=np.float))
    b = np.ndarray((1, 3,), buffer=np.array(basic_b, dtype=np.float))
    tot_input = tf.Variable(np.array([ a, b ]))
    multiplier = tf.Constant([2.0])
    # multiply_operation = tf.Variable( input_pl * 2)
    multiply_operation = tf.mul( input_pl, multiplier)

    with tf.Session() as session:
        result = session.run(multiply_operation, feed_dict={input_pl: tot_input})
        print result


def example2():
    shape1 = [ .1,.2,.3 ]
    shape2 = [  [ .11,.12,.13 ],
                [ .21,.22,.23 ],
                [ .31,.32,.33 ]]


    var_shape1 = tf.Variable(shape1, name="var_shape_1")
    var_shape2 = tf.Variable(shape2, name="var_shape_2")
    init_op = tf.initialize_all_variables()


    with tf.Session() as session:
        session.run(init_op)
        print var_shape1.eval()
        print var_shape1.get_shape()
        print var_shape2.eval()
        print var_shape2.get_shape()


    shape3 = [
    [ [ .111,.112,.113 ], [ .121,.122,.123 ], [ .131,.132,.133 ] ],
    [ [ .211,.212,.213 ], [ .221,.222,.223 ] ]
    ]

    var_shape3 = tf.Variable(shape3, name="var_shape_3")
    with tf.Session() as session:
        session.run(tf.initialize_variables([var_shape3]))
        print var_shape3.eval()
        print var_shape3.get_shape()


# example()
example2()