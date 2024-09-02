import tensorflow as tf


class Config:
	language = 'Geo'
	e1 = 'data/' + language + '/ent_ids_1'
	e2 = 'data/' + language + '/ent_ids_2'
	ill = 'data/' + language + '/ref_ent_ids'
	kg1 = 'data/' + language + '/triples_1'
	kg2 = 'data/' + language + '/triples_2'
	model_path = 'include/model/checkpoints'
	epochs = 600
	dim = 200
	act_func = tf.nn.relu
	alpha = 0.1
	beta = 0.1
	gamma = 1.0
	k = 125
