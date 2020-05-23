import tensorflow as tf

class Disout(tf.keras.layers.Layer):
    '''
    disout
    论文：https://arxiv.org/abs/2002.11022
    '''
    def __init__(self, dist_prob, block_size=6, alpha=1.0, **kwargs):
        super(Disout, self).__init__(**kwargs)
        self.dist_prob = dist_prob      
        self.weight_behind = None
  
        self.alpha = alpha
        self.block_size = block_size

    def build(self, input_shape):
        pass
    
    @tf.function
    def call(self, x):
        
        if not self.trainable: 
            return x
        else:
            # tf.print('x:',tf.rank(x))
            if tf.rank(x) == 4:
                # 复制
                # x=tf.identity(x)
                # (batch_size,w,h,c)
                x_shape = tf.shape(x)
                # tf.print('x_shape:', x_shape)
                width = x_shape[1]
                height = x_shape[2]
                
                seed_drop_rate = self.dist_prob* tf.cast(width*height, tf.float32) / self.block_size**2 / tf.cast(( width -self.block_size + 1)*( height -self.block_size + 1), tf.float32)
                # tf.print('seed_drop_rate:', seed_drop_rate)
                # (w, h)，块状扰动中心区域
                valid_block_center = tf.zeros((width, height), dtype=x.dtype)
                            
                def numpy_func1(x, width, height):
                    x[int(self.block_size // 2):(width - (self.block_size - 1) // 2),int(self.block_size // 2):(height - (self.block_size - 1) // 2)]=1.0
                    return x
                valid_block_center = tf.numpy_function(numpy_func1, [valid_block_center, width, height], tf.float32)
                # (1, w, h, 1)
                valid_block_center = tf.expand_dims(tf.expand_dims(valid_block_center, axis=0), axis=-1)
                # (batch_size,w,h,c)
                randdist = tf.random.normal(x_shape, dtype=x.dtype)
                # (batch_size,w,h,c)
                block_pattern = tf.cast(((1 -valid_block_center + (1 - seed_drop_rate) + randdist) >= 1), dtype=tf.float32)
                

                if self.block_size == width and self.block_size == height:
                    # (batch_size,1,1,c)           
                    block_pattern = tf.expand_dims(tf.expand_dims(tf.math.reduce_min(tf.reshape(block_pattern,(x_shape[0],x_shape[1]*x_shape[2],x_shape[3])), axis=1), axis=-2), axis=-2)
                    # tf.print('block_pattern1:', tf.shape(block_pattern))
                else:
                    # (batch_size,w,h,c)
                    block_pattern = -tf.nn.max_pool2d(input=-block_pattern, ksize=(self.block_size, self.block_size), strides=(1, 1), padding='SAME')
                    # tf.print('block_pattern2:', tf.shape(block_pattern))
                # if self.block_size % 2 == 0:
                #     block_pattern = block_pattern[:, :-1, :-1, :]
                #     tf.print('block_pattern3:', tf.shape(block_pattern))

                percent_ones = tf.reduce_sum(block_pattern) / tf.math.reduce_prod(tf.cast(tf.shape(block_pattern),tf.float32))

                if not (self.weight_behind is None) and not(len(self.weight_behind)==0):
                    # self.weight_behind[0]：(kw,kh,in,out)
                    wtsize = tf.shape(self.weight_behind[0])[0]
                    # (kw,kh,in,1)
                    weight_max = tf.math.reduce_max(self.weight_behind[0], axis=-1, keepdims=True)
                    sig = tf.ones(tf.shape(weight_max),dtype=weight_max.dtype)
                    sig_mask = tf.cast(tf.random.normal(tf.shape(weight_max),dtype=sig.dtype)<0.5,dtype=tf.float32)
                    sig = sig * (1 - sig_mask) - sig_mask
                    weight_max = weight_max * sig 
                    weight_mean = tf.math.reduce_mean(weight_max, axis=(0,1), keepdims=True)
                    if wtsize == 1:
                        weight_mean = 0.1 * weight_mean
                    #print(weight_mean)
                mean=tf.math.reduce_mean(x)
                var=tf.math.reduce_variance(x)

                if not (self.weight_behind is None) and not(len(self.weight_behind)==0):
                    dist=self.alpha*weight_mean*(var**0.5)*tf.random.normal(x_shape, dtype=x.dtype)
                else:
                    dist=self.alpha*0.01*(var**0.5)*tf.random.normal(x_shape, dtype=x.dtype)
                # tf.print('block_pattern:', block_pattern)
                x=x*block_pattern
                dist=dist*(1-block_pattern)
                x=x+dist
                x=x/percent_ones
                # tf.print('x:', tf.shape(x))
                # tf.print('x2:', tf.shape(x2))
                return x
            else:
                return x

    def compute_output_shape(self, input_shape):
        '''计算输出shape'''
        return input_shape

