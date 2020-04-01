import tensorflow as tf


class Linear():

    def __init__(self, create_weights):
        self.create_weights = create_weights

    def forward(self, inputs):
        logits = self._fully_connected(inputs, out_dim=1)
        # logits = tf.squeeze(logits)
        return logits

    def _fully_connected(self, x, out_dim):
        """Fully Connected layer for final output."""
        x = tf.reshape(x, [-1, 2])
        if self.create_weights:
            w = tf.get_variable('affine', [x.get_shape()[1], out_dim],
                                initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
            b = tf.get_variable('bias', [out_dim], initializer=tf.constant_initializer())
        else:  # fast weights have already been created and just need to plug them in
            graph = tf.get_default_graph()
            w = graph.get_tensor_by_name(graph.get_name_scope() + '/affine:0')
            b = graph.get_tensor_by_name(graph.get_name_scope() + '/bias:0')
        return tf.nn.xw_plus_b(x, w, b)
