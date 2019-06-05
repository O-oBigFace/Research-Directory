# Seq2Seq 简单实现

>   献给学不会的你。

## 需要反复琢磨的内容

* tf.raw_rnn()的loop_fn
* 各种向量的维度





---

```python
import numpy as np # matrix math
import tensorflow as tf # ML
import helpers # formatting data, and generating random sequence data
tf.reset_default_graph()
sess = tf.InteractiveSession()
```


```python
PAD = 0
EOS = 1 

# 词汇表大小
vocab_size = 10
input_embedding_size = 20 # character length

encoder_hidden_units = 20 
decoder_hidden_units = encoder_hidden_units * 2 # 在原始的论文中，编码器、解码器隐层神经元个数一致。但是这里解码器的神经元个数为什么是编码器的两倍？
```


```python
# placeholders
# 编码器输入的形状为 encoder_max_len * batch_size
encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name="encoder_inputs")

# 指定输入到编码器的序列的长度，这里将会用padding把所有的序列变为相等的长度。
# 如果不希望padding，则可以用dynamic memory network来输入可变长度的序列
encoder_inputs_length = tf.placeholder(shape=(None, ), dtype=tf.int32, name="encoder_inputs_length")

# 解码器输出单元长度为 output_max_len * batch_size
decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name="decoder_targets")
```


```python
# embeddings
# 手工创建词汇表的向量嵌入
embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1, dtype=tf.float32))

# 通过词汇的id在词嵌入中寻找特征向量
encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
```

    WARNING:tensorflow:From /anaconda3/lib/python3.7/site-packages/tensorflow/python/framework/op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Colocations handled automatically by placer.

# encoder: Bi-LSTM


```python
# define encoder 
from tensorflow.python.ops.rnn_cell import LSTMCell, LSTMStateTuple
```


```python
encoder_cell = LSTMCell(encoder_hidden_units)
```

## 构建一个动态的双向循环神经网络
### tf.nn.bidirectional_dynamic_rnn
#### 输入
* 构建独立的正向和反向RNN，input_size和正向/反向的单元数目必须一致；
* 两个方向的初始状态均默认为0 (可通过参数传入改变)
* 没有**中间层状态被**返回
* 网络根据传入的sequence_length来完全展开，若sequence_length未给出，则网络将完全展开。
* time_major: 规定了inputs和outputs的格式。
    * 默认值为False，张量的形状为[batch_size, max_time, depth];
    * 若值为True，张量的形状必须为[max_time, batch_size, depth].

#### 输出
* 输出的整体格式是一个元组(outputs, output_states)
    * outputs: 同样是元组(output_fw, output_bw)
        * 若time_major == True: output的shape为 [max_time, batch_size, output_size]
    * output_states: 元组(output_state_fw, output_state_bw), 包含了正\反向最后一个单元的状态，
        * shape为[batch_size, unit_size]
    


```python

((encoder_fw_outputs, 
  encoder_bw_outputs),
 (encoder_fw_final_state,
  encoder_bw_final_state)) = (
    tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                    cell_bw=encoder_cell,
                                    inputs=encoder_inputs_embedded,
                                    sequence_length=encoder_inputs_length,
                                    dtype=tf.float32,
                                    time_major=True)
)

```

    WARNING:tensorflow:From <ipython-input-8-51634fcda1c0>:10: bidirectional_dynamic_rnn (from tensorflow.python.ops.rnn) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use `keras.layers.Bidirectional(keras.layers.RNN(cell))`, which is equivalent to this API
    WARNING:tensorflow:From /anaconda3/lib/python3.7/site-packages/tensorflow/python/ops/rnn.py:443: dynamic_rnn (from tensorflow.python.ops.rnn) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use `keras.layers.RNN(cell)`, which is equivalent to this API
    WARNING:tensorflow:From /anaconda3/lib/python3.7/site-packages/tensorflow/python/ops/rnn.py:626: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.cast instead.


### 输出整理
* 本应用中抛弃了RNN的输出，利用状态信息
* 在output_states中，包含了隐含层的两种输出：memory cell 和 hidden state output


```python
# bidirectional step
# hidden step
encoder_final_state_c = tf.concat((encoder_fw_final_state.c, encoder_bw_final_state.c), 1)

# output
encoder_final_state_h = tf.concat((encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

# TF Tuple used bt LSTM Cells for state_size, zero_state, and output state
encoder_final_state = LSTMStateTuple(c=encoder_final_state_c, h=encoder_final_state_h)
```




    LSTMStateTuple(c=<tf.Tensor 'bidirectional_rnn/fw/fw/while/Exit_3:0' shape=(?, 20) dtype=float32>, h=<tf.Tensor 'bidirectional_rnn/fw/fw/while/Exit_4:0' shape=(?, 20) dtype=float32>)



# decoder


```python
# defining decoder
decoder_cell = LSTMCell(decoder_hidden_units)
```

time和batch都是动态的，可在运行时改变。
在解码时，将上一个时间生成的token当做这个时间的输出为模型增添了鲁棒性。
利用导师驱动过程能使训练速度提高。
最佳时间是在训练过程中将它们随机混合。


```python
# 仅仅查看，并不需要
# 序列长度和batch_size都可以动态变化
encoder_max_time, batch_size = tf.unstack(tf.shape(encoder_inputs))
```


```python
# longer than encoder, we have two direction states, plus end sentence token
decoder_lengths = encoder_inputs_length + 3
decoder_lengths
```




    <tf.Tensor 'add:0' shape=(?,) dtype=int32>



### output projection
* output(t) -> output projection -> projection(t) (argmax) -> input embedding(t+1) -> input(t+1)


```python
# output projection
# define our weighs and biases

W = tf.Variable(tf.random_uniform([decoder_hidden_units, vocab_size], -1 , 1), dtype=tf.float32)
b = tf.Variable(tf.zeros([vocab_size]), dtype=tf.float32)
```

### decoder via tf.nn.raw_rnn


```python
# create padded inputs for the decoder from the word embeddings

assert EOS == 1 and PAD == 0

eos_time_slice = tf.ones([batch_size], dtype=tf.int32, name="EOS")
pad_time_slice = tf.zeros([batch_size], dtype=tf.int32, name="PAD")

# retrives rows of the params tensor. The behavior is similar using indexing with arrays in numpy
eos_step_embedded = tf.nn.embedding_lookup(embeddings, eos_time_slice)
pad_step_embedded = tf.nn.embedding_lookup(embeddings, pad_time_slice)
```


```python
# manually specifying loop function through time - to get initial cell state and input to RNN
# normally we'd just use dynamic_rnn, but lets get detailed here with raw_rnn

# 仅仅定义并返回这些值，并没有对它们进行操作
def loop_fn_initial():
    initial_elements_finished = (0>=decoder_lengths) # 针对于batchsize: all False at the initial step
    # end of sentence
    initial_input = eos_step_embedded
    # last time steps cell state
    initial_cell_state = encoder_final_state
    # none
    initial_cell_output = None
    # none
    initial_loop_state = None # we don't need to pass any additional information
    return (initial_elements_finished,
            initial_input,
            initial_cell_state,
            initial_cell_output,
            initial_loop_state
           )
```


```python
# attention mechanism --choose which previously generated token to pass as input in the next time
def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):
    
    def get_next_input():
        # dot product between previous output , then + biases
        # 始终利用encoder_final_state计算attention值，和以往看到的不同
        output_logits = tf.add(tf.matmul(previous_output, W), b) 
        # Logits simply means that the fuction operates on the unscaled output of
        # earlier layer and that the relative scale to understand the units is linear.
        # It means, in particular , the sum of the inputs may not equal 1, that the values are not probability values
        # (you might have an input of 5).
        # prediction value at current time step
        
        # Returns the index with the largest value across axes of a tensor.
        prediction = tf.argmax(output_logits, axis=1)
        # embed prediction for the next input 
        next_input = tf.nn.embedding_lookup(embeddings, prediction)
        return next_input

    elements_finished = (time >= decoder_lengths) # this operation produces boolean tensor of [batch_size]
                                                  # defining if corresponding sequence has ended.
    
    # 计算某个维度上数值的"逻辑与"
    finished = tf.reduce_all(elements_finished) # --> boolean scalar
    # Return either fn1() or fn2 based on the boolean predicate pred.
    input = tf.cond(finished, lambda:pad_step_embedded, get_next_input)
    
    # set previous to current 为什么？ 当前的这些值如何更新？
    state = previous_state
    output = previous_output
    loop_state = None
    
    return (elements_finished,
            input,
            state,
            output, 
            loop_state
           )

```


```python
def loop_fn(time, previous_output, previous_state, previous_loop_state):
    if previous_loop_state is None: # time == 0
        assert previous_loop_state is None and previous_state is None
        return loop_fn_initial()
    else:
        return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)

decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
decoder_outputs = decoder_outputs_ta.stack()
```


```python
decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
decoder_logits_flats = tf.add(tf.matmul(decoder_outputs_flat, W), b)

decoder_logits = tf.reshape(decoder_logits_flats, (decoder_max_steps, decoder_batch_size, vocab_size))
```


```python
decoder_prediction = tf.math.argmax(decoder_logits, 2)
```


```python
stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
    labels= tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32), 
    logits=decoder_logits
)

loss = tf.reduce_mean(stepwise_cross_entropy)

train_op = tf.train.AdamOptimizer().minimize(loss=loss)
```


```python
sess.run(tf.global_variables_initializer())
```


```python
batch_size = 100

batches = helpers.random_sequences(length_from=3, length_to=8,
                                  vocab_lower=2, vocab_upper=10,
                                  batch_size=batch_size)

```


```python
def next_feed():
    batch = next(batches)
    encoder_inputs_, encoder_input_lengths_ = helpers.batch(batch)
    decoder_targets_, _ = helpers.batch(
    [(sequence) + [EOS] + [PAD] * 2 for sequence in batch])
    
    return {
        encoder_inputs: encoder_inputs_,
        encoder_inputs_length: encoder_input_lengths_,
        decoder_targets: decoder_targets_,
    }
```


```python
loss_track = []
```


```python
max_batches = 3001 
batches_in_epoches = 1000 

try:
    for batch in range(max_batches):
        fd = next_feed()
        _, l = sess.run([train_op, loss], fd)
        loss_track.append(l)
        
        if batch == 0 or batch % batches_in_epoches == 0:
            print(f"batch {batch}")
            print(f"    minibatch loss: {sess.run(loss, fd)}")
            predict_ = sess.run(decoder_prediction, fd)
            for i, (inp, pred) in enumerate(zip(fd[encoder_inputs].T, predict_.T)):
                print(f"    sample {i+1}")
                print(f"       input     > {inp}")
                print(f"       predicted > {pred}")
                if i >= 2:
                    break
            print()
            
except KeyboardInterrupt:
    print("training interrupted")
                
        
```

    batch 0
        minibatch loss: 2.3034567832946777
        sample 1
           input     > [4 9 4 4 3 0 0 0]
           predicted > [5 4 0 0 0 8 8 8 0 0 0]
        sample 2
           input     > [8 2 5 6 3 4 8 8]
           predicted > [2 2 4 8 8 8 9 4 1 1 7]
        sample 3
           input     > [3 5 4 4 5 8 3 4]
           predicted > [5 4 5 4 0 0 0 0 0 8 8]
    
    batch 1000
        minibatch loss: 0.5692545175552368
        sample 1
           input     > [4 8 7 7 8 2 8 5]
           predicted > [4 8 8 7 8 8 5 5 1 0 0]
        sample 2
           input     > [4 4 3 8 3 8 4 5]
           predicted > [4 4 3 8 3 4 5 5 1 0 0]
        sample 3
           input     > [5 9 2 9 3 5 3 3]
           predicted > [5 9 9 2 3 3 3 3 1 0 0]
    
    batch 2000
        minibatch loss: 0.23446983098983765
        sample 1
           input     > [4 8 5 7 0 0 0 0]
           predicted > [4 8 5 7 1 0 0 0 0 0 0]
        sample 2
           input     > [5 2 2 0 0 0 0 0]
           predicted > [5 2 2 1 0 0 0 0 0 0 0]
        sample 3
           input     > [5 5 3 0 0 0 0 0]
           predicted > [5 5 3 1 0 0 0 0 0 0 0]
    
    batch 3000
        minibatch loss: 0.13233636319637299
        sample 1
           input     > [4 6 9 9 0 0 0 0]
           predicted > [4 6 9 9 1 0 0 0 0 0 0]
        sample 2
           input     > [6 7 5 4 4 0 0 0]
           predicted > [6 7 5 4 4 1 0 0 0 0 0]
        sample 3
           input     > [5 4 9 0 0 0 0 0]
           predicted > [5 4 9 1 0 0 0 0 0 0 0]




```python
%matplotlib inline
import matplotlib.pyplot as plt
plt.plot(loss_track)
print('loss {:.4f} after {} examples (batch_size={})'.format(loss_track[-1], len(loss_track)*batch_size, batch_size))
```

    loss 0.1313 after 300100 examples (batch_size=100)



![out](/Users/bigface/Documents/markdown/Seq2seq简单实现/img/output_33_1.png)

