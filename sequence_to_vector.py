# std lib imports
from typing import Dict

# external libs
import tensorflow as tf
from tensorflow.keras import layers, models


class SequenceToVector(models.Model):
    """
    It is an abstract class defining SequenceToVector enocoder
    abstraction. To build you own SequenceToVector encoder, subclass
    this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    """
    def __init__(self,
                 input_dim: int) -> 'SequenceToVector':
        super(SequenceToVector, self).__init__()
        self._input_dim = input_dim
        

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> Dict[str, tf.Tensor]:
        """
        Forward pass of Main Classifier.

        Parameters
        ----------
        vector_sequence : ``tf.Tensor``
            Sequence of embedded vectors of shape (batch_size, max_tokens_num, embedding_dim)
        sequence_mask : ``tf.Tensor``
            Boolean tensor of shape (batch_size, max_tokens_num). Entries with 1 indicate that
            token is a real token, and 0 indicate that it's a padding token.
        training : ``bool``
            Whether this call is in training mode or prediction mode.
            This flag is useful while applying dropout because dropout should
            only be applied during training.

        Returns
        -------
        An output dictionary consisting of:
        combined_vector : tf.Tensor
            A tensor of shape ``(batch_size, embedding_dim)`` representing vector
            compressed from sequence of vectors.
        layer_representations : tf.Tensor
            A tensor of shape ``(batch_size, num_layers, embedding_dim)``.
            For each layer, you typically have (batch_size, embedding_dim) combined
            vectors. This is a stack of them.
        """
        # ...
        # return {"combined_vector": combined_vector,
        #         "layer_representations": layer_representations}
        raise NotImplementedError


class DanSequenceToVector(SequenceToVector):
    """
    It is a class defining Deep Averaging Network based Sequence to Vector
    encoder. You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this DAN encoder.
    dropout : `float`
        Token dropout probability as described in the paper.
    """
    def __init__(self, input_dim: int, num_layers: int, dropout: float = 0.2):
        super(DanSequenceToVector, self).__init__(input_dim)
        # TODO(students): start
        # ...
        # TODO(students): end
        self.listvar = []
        self._input_dim=input_dim
        self.dropout= dropout
        self.num_layers=num_layers
        for _ in range(num_layers): 
            var=tf.keras.layers.Dense(input_dim, activation="relu")
            self.listvar.append(var)
        
        

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> tf.Tensor:
        # TODO(students): start    
        sequence_mask= tf.cast(sequence_mask,tf.float32)
        dimensions= vector_sequence.get_shape().as_list()   #get dimensions
        if(training == True): #dropping words from the sentence
            arr= tf.random.uniform(tf.shape(sequence_mask))
            boolarr= arr>self.dropout
            boolarr= tf.cast(boolarr,tf.float32)
            sequence_mask= boolarr*sequence_mask
            
        seqmask2d= tf.tile(sequence_mask,[1,dimensions[2]])
        matrix3d=tf.reshape(seqmask2d,[dimensions[0],dimensions[2],dimensions[1]])
        trans= tf.transpose(matrix3d,perm=[0,2,1])
        vector_sequence = vector_sequence*trans
        vector_sequence_sum = tf.reduce_sum(vector_sequence,axis=1)
        
        actual_words = tf.reshape(tf.reduce_sum(sequence_mask,axis=1),[-1,1]) #find all 1's- will be one column
        vector_sequence_sum = tf.reduce_sum(vector_sequence,axis=1)
        vector_sequence_avg = vector_sequence_sum/actual_words
        list_layers=[]
        list_layers.append(self.listvar[0](vector_sequence_avg))
        for i in range(1,self.num_layers):
            list_layers.append(self.listvar[i](list_layers[i-1]))

        layer_representations=tf.transpose(tf.stack(list_layers),[1,0,2])
        combined_vector=list_layers[-1]
        # TODO(students): end
        return {"combined_vector": combined_vector,
                "layer_representations": layer_representations}
     
        


class GruSequenceToVector(SequenceToVector):
    """
    It is a class defining GRU based Sequence To Vector encoder.
    You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this GRU-based encoder. Note that each layer
        is a GRU encoder unlike DAN where they were feedforward based.
    """
    def __init__(self, input_dim: int, num_layers: int):
        super(GruSequenceToVector, self).__init__(input_dim)
        self.listvar = []
        self.input_dim=input_dim
        self.num_layers=num_layers
        for _ in range(num_layers): 
            var=tf.keras.layers.GRU(input_dim,return_sequences=True,return_state=True)
            self.listvar.append(var)

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> tf.Tensor:
        # TODO(students): start
         list_layers=[]
         outlayers=[]
         output,state= self.listvar[0](vector_sequence,mask=sequence_mask)
         outlayers.append(output)
         list_layers.append(state)
         for i in range(1,self.num_layers):
            output,state= self.listvar[i](outlayers[-1])
            outlayers.append(output)
            list_layers.append(state)
         layer_representations=tf.transpose(tf.stack(list_layers),[1,0,2])
         combined_vector=list_layers[-1]
        # TODO(students): end
         return {"combined_vector": combined_vector,
                "layer_representations": layer_representations}
0