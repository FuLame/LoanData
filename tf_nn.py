def evaluate(train_x, train_y, test_x, test_y, beta_param, n):
    '''
    Creates and fits a Neural Network model
    :param train_x:
    :param train_y:
    :param test_x:
    :param test_y:
    :param beta_param: regularization parameter
    :param n: size of hidden layer
    :return: probabilities of each class
    '''
    import tensorflow as tf
    import numpy as np
    from sklearn.metrics import roc_auc_score

    #If need convert the horizontal vector test_y into vertical
    try: test_y = np.reshape(test_y.values, (len(test_y), 1))
    except: test_y = np.reshape(test_y, (len(test_y), 1))

    n_rows, n_cols = train_x.shape
    n_nodes_hl1 = n
    n_nodes_hl2 = n
    n_nodes_hl3 = n
    n_classes = 1

    batch_size = 1000
    hm_epochs = 1200
    keep_prob = 1.0
    beta = beta_param

    x = tf.placeholder('float', [None, n_cols])
    y = tf.placeholder('float')

    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([n_cols, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes])), }

    def neural_network_model(data):
        data = tf.nn.dropout(data, 1.0)

        l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
        l1 = tf.nn.relu(l1)
        l1 = tf.nn.dropout(l1, keep_prob)

        l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
        l2 = tf.nn.relu(l2)
        l2 = tf.nn.dropout(l2, keep_prob)

        l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
        l3 = tf.nn.relu(l3)
        l3 = tf.nn.dropout(l3, keep_prob)

        output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

        return output


    def train_neural_network(x):
        prediction = neural_network_model(x)

        cost = tf.nn.l2_loss(prediction-y)
        regularizers = tf.nn.l2_loss(hidden_1_layer['weights'])+\
                       tf.nn.l2_loss(hidden_2_layer['weights'])+\
                       tf.nn.l2_loss(hidden_3_layer['weights'])

        #Cost function with regularization
        cost = tf.reduce_mean(cost + beta*regularizers)
        #Adam optimizer
        optimizer = tf.train.AdamOptimizer().minimize(cost)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(1,hm_epochs+1):
                epoch_loss = 0
                i = 0
                #Batch training
                while (i+batch_size) < n_rows:
                    start = i
                    end = i + batch_size

                    batch_x = np.array(train_x[start:end])
                    batch_y = np.array(train_y[start:end])
                    batch_y = np.reshape(batch_y, (len(batch_y), 1))

                    _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                    epoch_loss += c
                    i += batch_size

                #Every x epochs -> evaluate
                if epoch%1 == 0:
                    print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
                    #Calculate AUC scores for training and test set
                    #Those can be stored with loss value to obtain training graphs
                    prediction_sig_test = tf.nn.sigmoid(prediction).eval({x:test_x, y:test_y})
                    prediction_sig_train = tf.nn.sigmoid(prediction).eval(
                        {x: train_x, y: np.reshape(train_y, (len(train_y), 1))})

                    score_test = roc_auc_score(test_y, prediction_sig_test)
                    score_train = roc_auc_score(train_y, prediction_sig_train)

                    print("Train AUC:",score_train)
                    print("Test AUC:", score_test)
            #Sigmoid function is equivalent to softmax for one output neuron
            #Return probabilities
            prediction_sig = tf.nn.sigmoid(prediction).eval({x: test_x, y: test_y})
            return prediction_sig

    return train_neural_network(x)
