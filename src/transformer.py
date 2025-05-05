import tensorflow as tf  
import numpy as np

class DenseLayer():
    def __init__(self, input_dim, output_dim):
        d = tf.sqrt(tf.cast(input_dim, tf.float32))
        self.W = tf.Variable(tf.random.uniform([input_dim, output_dim], -1/d, 1/d))
        self.b = tf.Variable(tf.zeros([output_dim]))

        self.parameter_list = [self.W, self.b]

    def __call__(self, x):
        return tf.linalg.matmul(x, self.W) + self.b

    
class Transformer:
    def __init__(self, 
                 vocab_size, 
                 max_seq_len,
                 heads,
                 embed_dim,
                 key_dim,
                 ffnn_dims,
                 unembed_dims,
                 lr): 
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.heads = heads
        self.max_seq_len = max_seq_len
        self.key_dim = key_dim
        self.ffnn_dims = ffnn_dims
        self.unembed_dims = unembed_dims

        self.head_dim = embed_dim // heads

        d = tf.sqrt(tf.cast(self.embed_dim, tf.float32))

        self.word_embed = tf.Variable(tf.random.uniform([vocab_size, embed_dim], -1/d, 1/d))
        self.pos_embed = tf.Variable(tf.random.uniform([max_seq_len, embed_dim], -1/d, 1/d))

        self.WK =  tf.Variable(tf.random.uniform([heads, key_dim, embed_dim], -1/d, 1/d))
        self.WQ =  tf.Variable(tf.random.uniform([heads, key_dim, embed_dim], -1/d, 1/d))
        self.WV =  tf.Variable(tf.random.uniform([heads, self.head_dim, embed_dim], -1/d, 1/d))


        self.ffnn_dims.insert(0, embed_dim)
        self.ffnn_dims.append(embed_dim)
        self.ffnn_layers = []
        for i in range(len(ffnn_dims)-1):
             self.ffnn_layers.append(DenseLayer(ffnn_dims[i], ffnn_dims[i+1]))

        self.unembed_dims.insert(0, embed_dim)
        self.unembed_dims.append(vocab_size)   
        self.unembed_layers = []
        for i in range(len(unembed_dims)-1):
            self.unembed_layers.append(DenseLayer(unembed_dims[i], unembed_dims[i+1]))



        self.parameter_list = [self.word_embed, self.pos_embed, 
                               self.WK, self.WQ, self.WV]
        for layer in self.ffnn_layers:
            self.parameter_list += layer.parameter_list
        for layer in self.unembed_layers:
            self.parameter_list += layer.parameter_list
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def pred(self, x):

        x_embeds = self.embed(x)
        x_embeds = self.attention(x_embeds)
        x_embeds = self.ffnn(x_embeds)
        y_pred = self.unembed(x_embeds)

        return y_pred
    
    
    def embed(self, x):
        seq = tf.shape(x)[1]
        if seq > self.max_seq_len:
            x = x[-self.max_seq_len:]
            seq = self.max_seq_len
        x_embeds = tf.nn.embedding_lookup(self.word_embed, x)
        x_embeds = x_embeds + tf.expand_dims(self.pos_embed[:seq], axis=0)

        return x_embeds
    

    def attention(self, x_embeds):
        batch = tf.shape(x_embeds)[0]
        seq = tf.shape(x_embeds)[1]

        x_k = tf.einsum('ikl, bjl -> bijk', self.WK, x_embeds)
        x_q = tf.einsum('ikl, bjl -> bijk', self.WQ, x_embeds)
        x_v = tf.einsum('ikl, bjl -> bijk', self.WV, x_embeds)

        inner = tf.einsum('bijl,bikl -> bijk', x_k, x_q)
        mask = tf.linalg.band_part(tf.ones((1, seq, seq), dtype = tf.bool), -1, 0)
        mask = tf.repeat(mask, self.heads, axis=0)

        inner_masked = tf.where(mask, inner, tf.constant(-np.inf))

        dk = tf.sqrt(tf.cast(self.key_dim, tf.float32))
        WA = tf.nn.softmax(inner_masked/dk, axis=-1)

        head_outs = WA @ x_v
        concat   = tf.transpose(head_outs, [0, 2, 1, 3])  # [batch, seq, heads, head_dim]
        out   = tf.reshape(concat, [batch, seq, self.embed_dim])
     
        x_embeds = x_embeds + out

        return x_embeds
    

    def ffnn(self, x_embeds):
        x_up = x_embeds
        for layer in self.ffnn_layers[:-1]:
            x_up = layer(x_up)
            x_up = tf.nn.relu(x_up)
        x_down = self.ffnn_layers[-1](x_up)

        x_embeds = x_embeds + x_down
        
        return x_embeds
        
    
    def unembed(self, x_embeds):
        for layer in self.unembed_layers[:-1]:
            x_embeds = layer(x_embeds)
            x_embeds = tf.nn.relu(x_embeds)
        
        x_embeds = self.unembed_layers[-1](x_embeds)
        y_pred = tf.nn.softmax(x_embeds, axis=-1)
        return y_pred

    
    @tf.function
    def train_step(self, indices, y_true):
        
        with tf.GradientTape() as tape:
            loss = self.evaluate(indices, y_true)

        grads = tape.gradient(loss, self.parameter_list)
        self.optimizer.apply_gradients(zip(grads, self.parameter_list))
        return loss

    def evaluate(self, indices, y_true):
        y_true = y_true[:, 1:]
        y_pred = self.pred(indices)[:,:-1]
        loss = CrossEntropyLoss(y_true, y_pred)
        return loss


    
def CrossEntropyLoss(y_true, y_pred):
    loss = -tf.reduce_mean(y_true * tf.math.log(y_pred + 1e-10))
    return loss