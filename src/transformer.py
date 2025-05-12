import tensorflow as tf
import numpy as np


class DenseLayer(tf.keras.Model):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        d_xavier = np.sqrt(6.0 / (input_dim + output_dim))
        self.W = tf.Variable(tf.random.uniform([input_dim, output_dim], -d_xavier , d_xavier))
        self.b = tf.Variable(tf.zeros([output_dim]))

        self.parameter_list = [self.W, self.b]

    def __call__(self, x):
        W = tf.cast(self.W, x.dtype)
        b = tf.cast(self.b, x.dtype)
        return tf.linalg.matmul(x, W) + b


class TransformerBlock(tf.keras.Model):
    def __init__(
        self,
        vocab_size,
        max_seq_len,
        heads,
        embed_dim,
        ff_dim,
        dropout,
        **kwargs

    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.heads = heads
        self.max_seq_len = max_seq_len
        self.ff_dim = ff_dim
        self.dropout = dropout

        self.head_dim = embed_dim // heads
        self.dol1 = tf.keras.layers.Dropout(dropout)
        self.dol2 = tf.keras.layers.Dropout(dropout)
        self.dol3 = tf.keras.layers.Dropout(dropout)

        d = tf.sqrt(tf.cast(self.embed_dim, tf.float32))

        self.KQV = tf.Variable(
            tf.random.uniform([embed_dim, 3*embed_dim], -1 / d, 1 / d), name="KQV"
        )
        self.WO = tf.Variable(
            tf.random.uniform([embed_dim, embed_dim], -1 / d, 1 / d), name="WO"
        )

        self.layer_up = DenseLayer(embed_dim, ff_dim)
        self.layer_down = DenseLayer(ff_dim, embed_dim)

        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln1.build((None, None, embed_dim))

        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln2.build((None, None, embed_dim))

        self.parameter_list = [
            self.KQV,
            self.WO,
        ]
        self.parameter_list += self.layer_up.parameter_list
        self.parameter_list += self.layer_down.parameter_list
        self.parameter_list += self.ln1.trainable_variables
        self.parameter_list += self.ln2.trainable_variables

        self.parameter_decay = [
            self.KQV,
            self.WO,
            self.layer_up.W,
            self.layer_down.W,
        ]

    def attention(self, x_embeds, training=False):
        batch = tf.shape(x_embeds)[0]
        seq = tf.shape(x_embeds)[1]

        kqv = tf.cast(self.KQV, x_embeds.dtype)
        wo = tf.cast(self.WO, x_embeds.dtype)

        x_kqv = tf.matmul(x_embeds, kqv)
        x_kqv = tf.reshape(x_kqv, [batch, seq, self.heads, 3, self.head_dim])
        x_kqv = tf.transpose(x_kqv, [0, 2, 3, 1, 4])
        x_k = x_kqv[:, :, 0, :, :]
        x_q = x_kqv[:, :, 1, :, :]
        x_v = x_kqv[:, :, 2, :, :]


        inner = tf.einsum("bijl, bikl -> bijk", x_q, x_k)
        mask = tf.linalg.band_part(tf.ones((1, 1, seq, seq), dtype=tf.bool), -1, 0)

        inner_masked = tf.where(mask, inner, tf.constant(-np.inf, x_embeds.dtype))
        

        dk = tf.sqrt(tf.cast(self.head_dim, x_embeds.dtype))
        WA = tf.nn.softmax(inner_masked / dk, axis=-1)
        WA = self.dol1(WA, training)

        head_outs = WA @ x_v
        concat = tf.transpose(head_outs, [0, 2, 1, 3])  # [batch, seq, heads, head_dim]
        out = tf.reshape(concat, [batch, seq, self.embed_dim])
        out = tf.einsum("ijk,kl -> ijl", out, wo)

        # pre-norm to keep gradients alive
        out = self.dol2(out)
        out = self.ln1(out)
        out = out + x_embeds

        return out
    
    def ffnn(self, x_embeds, training=False):
        out = self.layer_up(x_embeds)
        out = tf.nn.relu(out)
        out = self.layer_down(out)
        out = self.dol3(out, training)
        out = self.ln2(out)

        out = out + x_embeds

        return out
    
    def call(self, x_embeds, training=False):
        x_embeds = self.attention(x_embeds, training)
        x_embeds = self.ffnn(x_embeds, training)

        return x_embeds



class Transformer(tf.keras.Model):
    def __init__(
        self,
        vocab_size,
        max_seq_len,
        tf_blocks,
        embed_dim,
        heads,
        ff_dim,
        unembed_dims,
        lr,
        wd=None,
        dropout=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.heads = heads
        self.max_seq_len = max_seq_len
        self.tf_blocks = tf_blocks
        self.ff_dim = ff_dim
        self.unembed_dims = unembed_dims
        self.wd = wd
        self.dropout = dropout

        self.head_dim = embed_dim // heads
        self.dol = tf.keras.layers.Dropout(dropout)


        d = tf.sqrt(tf.cast(self.embed_dim, tf.float32))

        self.word_embed = tf.Variable(
            tf.random.uniform([vocab_size, embed_dim], -1 / d, 1 / d), name="W_embed"
        )
        self.pos_embed = tf.Variable(
            tf.random.uniform([max_seq_len, embed_dim], -1 / d, 1 / d), name="W_pos_embed"
        )

        self.tf_blocks = []
        for i in range(tf_blocks):
            self.tf_blocks.append(TransformerBlock(vocab_size, max_seq_len, heads, embed_dim, ff_dim, dropout))
    

        self.unembed_b = tf.Variable(tf.zeros([vocab_size]))
        self.parameter_list = [
            self.word_embed,
            self.pos_embed,
        ]

        for block in self.tf_blocks:
            self.parameter_list += block.parameter_list
        
        self.parameter_list.append(self.unembed_b)

        self.parameter_decay = [
            self.word_embed,
            self.pos_embed,]
        
        for block in self.tf_blocks:
            self.parameter_decay += block.parameter_decay

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.opt = tf.keras.mixed_precision.LossScaleOptimizer(self.optimizer, dynamic=True)

    def call(self, x, training=False, logits=True, ):

        x = self.embed(x, training)
        if training:
            x = tf.cast(x, tf.float16)
            
        for block in self.tf_blocks:
            x = block.call(x, training)
        
        if logits:
            x = self.unembed(x)

        return x
    
    def embed(self, x, training=False):
        seq = tf.shape(x)[1]
        if seq > self.max_seq_len:
            x = x[:, -self.max_seq_len :]
            seq = self.max_seq_len
        x_embeds = tf.nn.embedding_lookup(self.word_embed, x)
        x_embeds = x_embeds + tf.expand_dims(self.pos_embed[:seq], axis=0)
        x_embeds = self.dol(x_embeds, training)

        return x_embeds

    def unembed(self, x_embeds):
        w_embed = tf.cast(self.word_embed, x_embeds.dtype)
        unembed_b = tf.cast(self.unembed_b, x_embeds.dtype)

        logits = x_embeds @ tf.transpose(w_embed) + unembed_b

        return logits

    @tf.function()
    def train_step(self, indices, y_true):

        with tf.GradientTape() as tape:
            loss = self.evaluate(indices, y_true, training=True)
            scaled_loss = self.opt.get_scaled_loss(loss)

        scaled_grads = tape.gradient(scaled_loss, self.parameter_list)
        grads = self.opt.get_unscaled_gradients(scaled_grads)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)

        self.opt.apply_gradients(zip(grads, self.parameter_list))

        get_lr = self.optimizer._decayed_lr(tf.float32)
        
        if self.wd is not None:
            for param in self.parameter_decay:
                param.assign_sub(get_lr*self.wd * param)
                
        return loss

    def evaluate(self, indices, y_true, training = False):
        y_true = y_true[:, 1:]
        
        logits = self.call(indices[:, :-1], training)
        logits32 = tf.cast(logits, tf.float32)
        loss = tf.math.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_true, logits32, from_logits=True))
        return loss