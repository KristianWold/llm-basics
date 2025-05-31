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
        tokenizer,
        **kwargs

    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.heads = heads
        self.max_seq_len = max_seq_len
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.tokenizer = tokenizer

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


    def call(self, x_embeds, tokens, training=False):
        x_embeds = self.attention(x_embeds, tokens, training)
        x_embeds = self.ffnn(x_embeds, training)

        return x_embeds


    def attention(self, x_embeds, tokens, training=False):
        batch = tf.shape(x_embeds)[0]
        seq = tf.shape(x_embeds)[1]

        kqv = tf.cast(self.KQV, x_embeds.dtype)
        wo = tf.cast(self.WO, x_embeds.dtype)

        x_kqv = tf.matmul(x_embeds, kqv)
        x_kqv = tf.reshape(x_kqv, [batch, seq, self.heads, 3, self.head_dim])
        x_kqv = tf.transpose(x_kqv, [0, 2, 3, 1, 4])
        x_k = x_kqv[:, :, 0, :, :] # [batch, heads, seq, head_dim]
        x_q = x_kqv[:, :, 1, :, :] # [batch, heads, seq, head_dim]
        x_v = x_kqv[:, :, 2, :, :] # [batch, heads, seq, head_dim]

        x_k = tf.transpose(x_k, [0, 1, 3, 2])  # [batch, seq, head_dim, heads]
        inner = tf.matmul(x_q, x_k)
        mask = tf.linalg.band_part(tf.ones((1, 1, seq, seq), dtype=tf.bool), -1, 0)

        inner_masked = tf.where(mask, inner, tf.constant(-np.inf, x_embeds.dtype))

        if "<s>" in self.tokenizer.token_to_idx:
            inner_masked = put_block_diag_mask(inner_masked, tokens, self.tokenizer.token_to_idx["<s>"])
        

        dk = tf.sqrt(tf.cast(self.head_dim, x_embeds.dtype))
        WA = tf.nn.softmax(inner_masked / dk, axis=-1)
        WA = self.dol1(WA, training)

        head_outs = WA @ x_v
        concat = tf.transpose(head_outs, [0, 2, 1, 3])  # [batch, seq, heads, head_dim]
        out = tf.reshape(concat, [batch, seq, self.embed_dim])
        out = tf.matmul(out, wo)

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
        tokenizer,
        lr,
        wd=None,
        dropout=0.1,
        accum_steps = 0,
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
        self.tokenizer = tokenizer
        self.wd = wd
        self.dropout = dropout
        self.accum_steps = tf.constant(accum_steps, dtype=tf.int32)

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
            self.tf_blocks.append(TransformerBlock(vocab_size, max_seq_len, heads, embed_dim, ff_dim, dropout, tokenizer))
    

        self.unembed_b = tf.Variable(tf.zeros([vocab_size]))
        self.parameter_list = [
            self.word_embed,
            self.pos_embed,
        ]

        for block in self.tf_blocks:
            self.parameter_list += block.parameter_list
        
        self.parameter_list.append(self.unembed_b)

        # gradient accumulation

        self.step_counter = tf.Variable(0, trainable=False)

        self.accum_grads = []
        for param in self.parameter_list:
            self.accum_grads.append(tf.Variable(tf.zeros_like(param), trainable=False))

        # decay weights
        self.parameter_decay = [
            self.word_embed,
            self.pos_embed,]
        
        for block in self.tf_blocks:
            self.parameter_decay += block.parameter_decay

        self.opt = tf.keras.mixed_precision.LossScaleOptimizer(tf.keras.optimizers.Adam(learning_rate=lr), 
                                                               dynamic=True)
        
    def call(self, tokens, training=False, logits=True,):

        x, tokens = self.embed(tokens, training)
            
        for block in self.tf_blocks:
            x = block.call(x, tokens, training)
        
        if logits:
            x = self.unembed(x)

        return x
    
    def embed(self, tokens, training=False):
        seq = tf.shape(tokens)[1]
        if seq > self.max_seq_len:
            tokens = tokens[:, -self.max_seq_len :]
            seq = self.max_seq_len
        x_embeds = tf.nn.embedding_lookup(self.word_embed, tokens)
        if training:
            x_embeds = tf.cast(x_embeds, tf.float16)

        start_token = self.tokenizer.token_to_idx["<s>"]
        pos_idx = resetting_positions(tokens, start_token)
        pos_embed = tf.cast(tf.nn.embedding_lookup(self.pos_embed, pos_idx), x_embeds.dtype)

        x_embeds = x_embeds + pos_embed
        x_embeds = self.dol(x_embeds, training)

        return x_embeds, tokens

    def unembed(self, x_embeds):
        w_embed = tf.cast(self.word_embed, x_embeds.dtype)
        unembed_b = tf.cast(self.unembed_b, x_embeds.dtype)

        logits = x_embeds @ tf.transpose(w_embed) + unembed_b

        return logits

    @tf.function(input_signature=[tf.TensorSpec([None, None], tf.int32)])
    def train_step(self, tokens):
        print("🔄 Tracing train_step; token shape:", tokens.shape)

        get_lr = self.opt.inner_optimizer._decayed_lr("float32")

        with tf.GradientTape() as tape:
            loss = self.evaluate(tokens, training=True)
            scaled_loss = self.opt.get_scaled_loss(loss)

        scaled_grads = tape.gradient(scaled_loss, self.parameter_list)
        grads = self.opt.get_unscaled_gradients(scaled_grads)
        
        if self.accum_steps > 0:
            self.step_counter.assign_add(1)
            self.accumulate_gradients(grads)

            if tf.equal(self.step_counter, self.accum_steps):
                self.step_counter.assign(0)
                
                grads = [accum_grad / tf.cast(self.accum_steps, tf.float32) for accum_grad in self.accum_grads]
                grads, _ = tf.clip_by_global_norm(grads, 1.0)
                
                self.opt.apply_gradients(zip(grads, self.parameter_list))

                if self.wd is not None:
                    for param in self.parameter_decay:
                        param.assign_sub(get_lr*self.wd * param)

                for accum_grad in self.accum_grads:
                    accum_grad.assign(tf.zeros_like(accum_grad))
        else:
            grads, _ = tf.clip_by_global_norm(grads, 1.0)
            self.opt.apply_gradients(zip(grads, self.parameter_list))

            if self.wd is not None:
                for param in self.parameter_decay:
                    param.assign_sub(get_lr*self.wd * param)
        
        return loss

    def evaluate(self, tokens, training = False):
        y_true = tokens[:, 1:]
        
        logits = self.call(tokens[:, :-1], training)
        logits32 = tf.cast(logits, tf.float32)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, logits32, from_logits=True)

        if "<pad>" in self.tokenizer.token_to_idx:
            mask = tf.cast(tf.not_equal(y_true, self.tokenizer.token_to_idx["<pad>"]), tf.float32)
            loss = loss * mask
            loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
        else:
            loss = tf.reduce_mean(loss)

        return loss
    
    def accumulate_gradients(self, grads):
        for accum_grad, grad in zip(self.accum_grads, grads):
            accum_grad.assign_add(grad)
    
    def get_num_params(self):
        total_params = 0
        for var in self.parameter_decay:
            shape = var.get_shape()
            num_params = 1
            for dim in shape:
                num_params *= dim
            total_params += num_params

        return total_params

    def get_weight_norm(self):
        weight_norm = 0
        for param in self.parameter_decay:
            weight_norm += tf.reduce_sum(tf.abs(param))
        return weight_norm.numpy()/self.get_num_params()
    


def put_block_diag_mask(inner, tokens, start_token_id):

    is_start = tf.cast(tf.equal(tokens, start_token_id), tf.int32)
    segment_ids = tf.cumsum(is_start, axis=1)
    seg_i = tf.expand_dims(segment_ids, 2)                          
    seg_j = tf.expand_dims(segment_ids, 1)                         
    mask   = tf.equal(seg_i, seg_j)
    mask = tf.expand_dims(mask, 1)                                      # [B, 1, T, T]                             

    return tf.where(mask, inner, tf.constant(-np.inf, inner.dtype))


def resetting_positions(tokens, start_token_id):
    """
    tokens:          int32 Tensor of shape [batch, seq_len]
    start_token_id:  scalar int32 — the ID of your “start” token
    returns:
      rel_pos:       int32 Tensor of shape [batch, seq_len],
                     where rel_pos[b,i] counts up from 0 since
                     the last start‐token (or the sequence start).
    """
    # 1) detect start tokens
    is_start = tf.equal(tokens, start_token_id)                   # [B, T], bool

    # 2) force a “start” at position 0 of each sequence
    batch_size = tf.shape(tokens)[0]
    is_start = tf.concat([
        tf.ones([batch_size, 1], dtype=tf.bool),
        is_start[:, 1:]
    ], axis=1)                                                    # [B, T]

    # 3) make a [0,1,2,…,T-1] index array for each batch
    seq_len = tf.shape(tokens)[1]
    positions = tf.range(seq_len, dtype=tf.int32)                # [T]
    positions = tf.expand_dims(positions, 0)                      # [1, T]
    positions = tf.tile(positions, [batch_size, 1])              # [B, T]

    # 4) pick out the indices where resets happen (else 0)
    start_pos = tf.where(is_start, positions, tf.zeros_like(positions))  # [B, T]

    # 5) compute, for each token, the **latest** reset‐position seen so far.
    #    We need a cumulative‐maximum along axis=1.
    #    If you have TF 2.5+, you can do:
    try:
        last_start = tf.math.cummax(start_pos, axis=1)           # [B, T]
    except AttributeError:
        # fallback for older TF: use tf.scan over the time axis
        #  - transpose to [T, B], scan produces [T, B], then transpose back
        start_t = tf.transpose(start_pos, [1, 0])               # [T, B]
        init = tf.zeros([batch_size], dtype=start_pos.dtype)   # [B]
        cummax_t = tf.scan(
            lambda prev, cur: tf.maximum(prev, cur),
            elems=start_t,
            initializer=init
        )                                                       # [T, B]
        last_start = tf.transpose(cummax_t, [1, 0])             # [B, T]

    # 6) subtract to get “position since last reset”
    rel_pos = positions - last_start                            # [B, T]
    return rel_pos




class WarmUpThenDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self,
                 initial_learning_rate: float,
                 warmup_steps: int,
                 decay_schedule_fn: tf.keras.optimizers.schedules.LearningRateSchedule):
        """
        initial_learning_rate: peak LR reached at end of warmup
        warmup_steps:      # of steps to ramp from 0 → initial_learning_rate
        decay_schedule_fn: a tf.keras schedule to apply *after* warmup
        """
        super().__init__()
        self.initial_lr = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.decay_schedule_fn = decay_schedule_fn

    def __call__(self, step):
        # Cast to float32 for safety in graph mode
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)

        # compute linear warmup: lr = initial_lr * (step / warmup_steps)
        warmup_lr = self.initial_lr * (step / warmup_steps)

        # after warmup_steps, switch to decay schedule (shift step count)
        decay_step = step - warmup_steps
        decay_lr = self.decay_schedule_fn(decay_step)

        # if step < warmup_steps, pick warmup_lr, else decay_lr
        return tf.cond(step < warmup_steps,
                       lambda: warmup_lr,
                       lambda: decay_lr)