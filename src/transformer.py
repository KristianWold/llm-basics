import tensorflow as tf
import numpy as np


class DenseLayer(tf.keras.Model):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        d = tf.sqrt(tf.cast(input_dim, tf.float32))
        self.W = tf.Variable(tf.random.uniform([input_dim, output_dim], -1 / d, 1 / d))
        self.b = tf.Variable(tf.zeros([output_dim]))

        self.parameter_list = [self.W, self.b]

    def __call__(self, x):
        return tf.linalg.matmul(x, self.W) + self.b


class TransformerBlock(tf.keras.Model):
    def __init__(
        self,
        vocab_size,
        max_seq_len,
        heads,
        embed_dim,
        ff_dim,
        **kwargs

    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.heads = heads
        self.max_seq_len = max_seq_len
        self.ff_dim = ff_dim

        self.head_dim = embed_dim // heads

        d = tf.sqrt(tf.cast(self.head_dim, tf.float32))

        self.WK = tf.Variable(
            tf.random.uniform([heads, self.head_dim, embed_dim], -1 / d, 1 / d), name="WK"
        )
        self.WQ = tf.Variable(
            tf.random.uniform([heads, self.head_dim, embed_dim], -1 / d, 1 / d), name="WQ"
        )
        self.WV = tf.Variable(
            tf.random.uniform([heads, self.head_dim, embed_dim], -1 / d, 1 / d),
            name="WV",
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
            self.WK,
            self.WQ,
            self.WV,
            self.WO,
        ]
        self.parameter_list += self.layer_up.parameter_list
        self.parameter_list += self.layer_down.parameter_list
        self.parameter_list += self.ln1.trainable_variables
        self.parameter_list += self.ln2.trainable_variables

    def attention(self, x_embeds):
        batch = tf.shape(x_embeds)[0]
        seq = tf.shape(x_embeds)[1]

        x_k = tf.einsum("ikl, bjl -> bijk", self.WK, x_embeds)
        x_q = tf.einsum("ikl, bjl -> bijk", self.WQ, x_embeds)
        x_v = tf.einsum("ikl, bjl -> bijk", self.WV, x_embeds)

        inner = tf.einsum("bijl,bikl -> bijk", x_q, x_k)
        mask = tf.linalg.band_part(tf.ones((1, seq, seq), dtype=tf.bool), -1, 0)
        mask = tf.repeat(mask, self.heads, axis=0)

        inner_masked = tf.where(mask, inner, tf.constant(-np.inf))

        dk = tf.sqrt(tf.cast(self.head_dim, tf.float32))
        WA = tf.nn.softmax(inner_masked / dk, axis=-1)

        head_outs = WA @ x_v
        concat = tf.transpose(head_outs, [0, 2, 1, 3])  # [batch, seq, heads, head_dim]
        out = tf.reshape(concat, [batch, seq, self.embed_dim])
        out = tf.einsum("ijk,kl -> ijl", out, self.WO)

        # pre-norm to keep gradients alive
        out = self.ln1(out)
        out = out + x_embeds

        return out
    
    def ffnn(self, x_embeds):
        out = self.layer_up(x_embeds)
        out = tf.nn.relu(out)
        out = self.layer_down(out)
        out = self.ln2(out)
        out = out + x_embeds

        return out
    
    def call(self, x_embeds):
        x_embeds = self.attention(x_embeds)
        x_embeds = self.ffnn(x_embeds)

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

        self.head_dim = embed_dim // heads


        d = tf.sqrt(tf.cast(self.embed_dim, tf.float32))

        self.word_embed = tf.Variable(
            tf.random.uniform([vocab_size, embed_dim], -1 / d, 1 / d), name="word_embed"
        )
        self.pos_embed = tf.Variable(
            tf.random.uniform([max_seq_len, embed_dim], -1 / d, 1 / d), name="pos_embed"
        )

        self.tf_blocks = []
        for i in range(tf_blocks):
            self.tf_blocks.append(TransformerBlock(vocab_size, max_seq_len, heads, embed_dim, ff_dim))
    

        self.unembed_dims.insert(0, embed_dim)
        self.unembed_dims.append(vocab_size)
        self.unembed_layers = []
        for i in range(len(unembed_dims) - 1):
            self.unembed_layers.append(DenseLayer(unembed_dims[i], unembed_dims[i + 1]))


        self.parameter_list = [
            self.word_embed,
            self.pos_embed,
        ]

        for block in self.tf_blocks:
            self.parameter_list += block.parameter_list
        for layer in self.unembed_layers:
            self.parameter_list += layer.parameter_list

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def call(self, x):

        x_embeds = self.embed(x)
        for block in self.tf_blocks:
            x_embeds = block.call(x_embeds)
        
        y_pred = self.unembed(x_embeds)

        return y_pred

    def embed(self, x):
        seq = tf.shape(x)[1]
        if seq > self.max_seq_len:
            x = x[:, -self.max_seq_len :]
            seq = self.max_seq_len
        x_embeds = tf.nn.embedding_lookup(self.word_embed, x)
        x_embeds = x_embeds + tf.expand_dims(self.pos_embed[:seq], axis=0)

        return x_embeds

    def unembed(self, x_embeds):
        for layer in self.unembed_layers[:-1]:
            x_embeds = layer(x_embeds)
            x_embeds = tf.nn.relu(x_embeds)

        logits = self.unembed_layers[-1](x_embeds)

        return logits

    @tf.function()
    def train_step(self, indices, y_true):

        with tf.GradientTape() as tape:
            loss = self.evaluate(indices, y_true)

        grads = tape.gradient(loss, self.parameter_list)
        self.optimizer.apply_gradients(zip(grads, self.parameter_list))
        return loss

    def evaluate(self, indices, y_true):
        y_true = y_true[:, 1:]
        # categorical from index, not onehot encoding
        y_pred = self.call(indices)[:, :-1]
        loss = tf.math.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True))
        return loss

