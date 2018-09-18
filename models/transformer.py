from datetime import datetime
from typing import Optional
import tensorflow as tf
from mlcore.model.alphabot.utils import utils
from mlcore.tf import transformer


class PolicyNetwork:

    def __init__(self, config, scope='policy_network', reuse: bool=None) -> None:

        # instance var
        self.config = config
        self.scope = scope

        self.build_model(scope, reuse)
        self.init_global_step()
        self.init_saver()

    def init_saver(self):
        self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope))

    def init_global_step(self):
        self.global_step = tf.train.get_or_create_global_step()

    def increment_version(self, sess, date_string: str):
        '''
        :params sess: tensorflow session
        :params date_string: バージョンとなる日付
        '''
        assign_version = self.version.assign(date_string)
        sess.run(assign_version)

    def get_version(self, sess):
        '''
        外から叩かれてモデルのバージョンを返す。
        :params sess: tensorflow session
        '''
        return sess.run(self.version)

    def save(self, sess):
        self.saver.save(sess, self.config.checkpoint_path, self.global_step)

    def load(self, sess: tf.Session, path: str) -> None:
        '''
        モデルを読み込みます。
        :param sess: セッション
        :param path: モデルのパス。以下のいずれか
            - S3 上のファイルパス(s3://.../model.ckpt)
            - ローカルのディレクトリパス：ディレクトリをチェックポイントディレクトリとみなし
              最新のチェックポイントを読み込みます
            - ローカルのファイルパス： .../model.ckpt
        '''
        model_path = utils.get_model_path(path)
        if model_path:
            print("Loading model {} ...\n".format(model_path))
            self.saver.restore(sess, model_path)
            print("Model loaded")

    def build_model(self, scope, reuse: Optional[bool]):
        with tf.variable_scope(scope, reuse=reuse):
            # version
            self._define_version()

            # placeholder
            self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')
            self.encoder_inputs = tf.placeholder(
                dtype=tf.int32,
                shape=[None, self.config.max_length],
                name='encoder_inputs'
            )
            self.hier_encoder_inputs = tf.placeholder(
                dtype=tf.int32,
                shape=[None, self.config.max_context_length],
                name='hier_encoder_inputs'
            )
            self.decoder_targets = tf.placeholder(
                dtype=tf.int32,
                shape=[None, self.config.max_length],
                name='decoder_targets'
            )
            self.decoder_inputs = tf.placeholder(
                dtype=tf.int32,
                shape=[None, self.config.max_length],
                name='decoder_inputs'
            )

            # building
            sent_encoder_inputs_embedded = self._encoder()
            #hier_encoder_inputs_embedded = self._hier_encoder(sent_encoder_inputs_embedded)
            #self.decoder_logits = self._decoder(hier_encoder_inputs_embedded, self.decoder_inputs)
            self.decoder_logits = self._decoder(sent_encoder_inputs_embedded, self.decoder_inputs)

        is_target = tf.to_float(tf.not_equal(self.decoder_targets, 0))
        # loss
        decoder_targets_one_hot = tf.one_hot(self.decoder_targets, self.config.vocab_size)
        decoder_targets_smoothed = utils.label_smoothing(decoder_targets_one_hot)
        cross_ents = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.decoder_logits,
            labels=decoder_targets_smoothed
        )
        self.loss = tf.reduce_sum(cross_ents * is_target) / tf.reduce_sum(is_target)

        # acc
        predicted_ids = tf.to_int32(tf.argmax(self.decoder_logits, axis=2))
        correct = tf.equal(predicted_ids, self.decoder_targets)
        self.accuracy = tf.reduce_sum(tf.to_float(correct)*is_target) / (tf.reduce_sum(is_target))

    def _encoder(self):
        with tf.variable_scope('encoder'):
            encoder_inputs_embedded = transformer.embedding(
                self.encoder_inputs,
                self.config.vocab_size,
                self.config.num_hidden_units,
                is_scale=True,
                scope='enc_embed'
            )
            encoder_inputs_embedded += transformer.positional_encoding(
                self.encoder_inputs,
                num_units=self.config.num_hidden_units,
                is_zero_pad=True,
            )
            encoder_inputs_embedded = tf.layers.dropout(
                encoder_inputs_embedded,
                rate=self.config.dropout_rate,
                training=self.is_training
            )

            sent_encoder_init_state = tf.get_variable(  # [1, hidden_units]
                'sent_encoder_init_state',
                dtype=tf.float32,
                shape=[1, self.config.num_hidden_units],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            sent_encoder_inputs_embedded = tf.tile(
                tf.expand_dims(sent_encoder_init_state, 0),  # [1, 1, hidden_units]
                [tf.shape(encoder_inputs_embedded)[0], 1, 1]  # [batch_size, 1, hidden_units]
            )

            for i in range(self.config.num_blocks):
                with tf.variable_scope('block_{}'.format(i)):
                    encoder_inputs_embedded = transformer.multihead_attention(
                        queries=encoder_inputs_embedded,
                        keys=encoder_inputs_embedded,
                        is_training=self.is_training,
                        dropout_rate=self.config.dropout_rate,
                        num_units=self.config.num_hidden_units,
                        num_heads=self.config.num_heads,
                        is_causality=False
                    )

                    sent_encoder_inputs_embedded = transformer.multihead_attention(
                        queries=sent_encoder_inputs_embedded,
                        keys=encoder_inputs_embedded,
                        is_training=self.is_training,
                        dropout_rate=self.config.dropout_rate,
                        num_units=self.config.num_hidden_units,
                        num_heads=self.config.num_heads,
                        is_causality=False,
                        scope='hier_attention'
                    )
                    sent_encoder_inputs_embedded = transformer.feedforward(
                        sent_encoder_inputs_embedded,
                        num_units=[4*self.config.num_hidden_units, self.config.num_hidden_units],
                        scope='hier_feedforward'
                    )

            return sent_encoder_inputs_embedded

    def _hier_encoder(self, sent_encoder_inputs_embedded):
        with tf.variable_scope('hier_encoder'):
            sent_encoder_inputs_embedded = tf.squeeze(sent_encoder_inputs_embedded, 1)
            hier_encoder_inputs_lookup_table = tf.concat(
                [tf.zeros([1, self.config.num_hidden_units]), sent_encoder_inputs_embedded],
                axis=0
            )
            hier_encoder_inputs_embedded = tf.gather(hier_encoder_inputs_lookup_table, self.hier_encoder_inputs)
            hier_encoder_inputs_embedded += transformer.positional_encoding(
                self.hier_encoder_inputs,
                num_units=self.config.num_hidden_units,
                is_zero_pad=True,
            )

            for i in range(self.config.num_hier_blocks):
                with tf.variable_scope('blocks_{}'.format(i)):
                    hier_encoder_inputs_embedded = transformer.multihead_attention(
                        queries=hier_encoder_inputs_embedded,
                        keys=hier_encoder_inputs_embedded,
                        is_training=self.is_training,
                        dropout_rate=self.config.dropout_rate,
                        num_units=self.config.num_hidden_units,
                        num_heads=self.config.num_heads,
                        is_causality=False
                    )
                    hier_encoder_inputs_embedded = transformer.feedforward(
                        hier_encoder_inputs_embedded,
                        num_units=[4*self.config.num_hidden_units, self.config.num_hidden_units]
                    )

            return hier_encoder_inputs_embedded

    def _decoder(self, hier_encoder_inputs_embedded, decoder_inputs):
        with tf.variable_scope('decoder'):
            decoder_inputs_embedded = transformer.embedding(
                decoder_inputs,
                vocab_size=self.config.vocab_size,
                num_units=self.config.num_hidden_units,
                is_scale=True,
                scope='dec_embed'
            )
            decoder_inputs_embedded += transformer.positional_encoding(
                decoder_inputs,
                num_units=self.config.num_hidden_units,
                is_zero_pad=True,
            )

            for i in range(self.config.num_blocks):
                with tf.variable_scope('blocks_{}'.format(i)):
                    decoder_inputs_embedded = transformer.multihead_attention(
                        queries=decoder_inputs_embedded,
                        keys=decoder_inputs_embedded,
                        is_training=self.is_training,
                        dropout_rate=self.config.dropout_rate,
                        num_units=self.config.num_hidden_units,
                        num_heads=self.config.num_heads,
                        is_causality=True,
                        scope='self_attention'
                    )
                    decoder_inputs_embedded = transformer.multihead_attention(
                        queries=decoder_inputs_embedded,
                        keys=hier_encoder_inputs_embedded,
                        is_training=self.is_training,
                        dropout_rate=self.config.dropout_rate,
                        num_units=self.config.num_hidden_units,
                        num_heads=self.config.num_heads,
                        is_causality=False,
                        scope='vanilla_attention'
                    )

                    decoder_inputs_embedded = transformer.feedforward(
                        decoder_inputs_embedded,
                        num_units=[4*self.config.num_hidden_units, self.config.num_hidden_units]
                    )

            decoder_logits = tf.layers.dense(decoder_inputs_embedded, self.config.vocab_size)

            return decoder_logits

