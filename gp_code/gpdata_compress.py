import jax.numpy as np
from jax import grad


def compresion_loss(weights, kdd, y_train, ktd, targets):
    compressed_y_train = weights.T @ y_train
    compressed_ktd = ktd@weights
    compressed_kdd_inv =  np.linalg.inv(weights.T @ kdd @ weights)
    preds = compressed_ktd @ compressed_kdd_inv @ compressed_y_train
    return np.sum((preds - targets)**2)

def compress_x_train(kdd, y_train, kvd, y_val, ktd, y_test, n_com):
    n = kdd.shape[0]

    # Optimize weights using gradient descent.
    training_gradient_fun = grad(compresion_loss)
    weights = np.eye(n)
    weights = weights[:,:n_com]
    print("Initial loss:", compresion_loss(weights, kdd, y_train, kvd, y_val))
    for i in range(100):
        weights -= training_gradient_fun(weights, kdd, y_train, kvd, y_val) * 0.01 #gradient descent
    print("Trained loss:", compresion_loss(weights, kdd, y_train, kvd, y_val))

    com_y_train = weights.T @ y_train
    com_ktd = ktd@weights
    com_kdd = weights.T @ kdd @ weights  
    return weights, com_kdd, com_y_train, com_ktd


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    load_dct = True
    if load_dct:
        dct = np.load('KDD.npz')
        w = dct['w']
        v = dct['v']
        kdd = dct['kdd']
        ktd = dct['ktd']
        # kdd_re = np.dot(v, w[:,np.newaxis] * v.T)
        
        vit_feature_file = 'cifar_vit.npz'
        vit_dct = np.load(vit_feature_file)
        x_train_full = vit_dct['x_train']
        y_train_full = vit_dct['y_train']
        x_test = vit_dct['x_test']
        y_test = vit_dct['y_test']

        x_train_full = x_train_full[:20000]
        y_train_full = y_train_full[:20000]



    n = 1000
    n_com = 50
    kvd = kdd[n:,:][:,:n]
    y_val = y_train_full[n:]
    kdd = kdd[:,:n][:n,:]
    y_train_full = y_train_full[:n]
    ktd = ktd[:,:n]

    # Original inference
    kdd_inv = np.linalg.inv(kdd)
    mu = ktd@kdd_inv@y_train_full
    acc = sum(np.argmax(y_test,axis=1) == np.argmax(mu,axis=1)) / len(mu)

    com_accs = []
    ns_com = [10,50,100,200,300,400,500,600,700,800,900,1000]
    for n_com in ns_com:
        weights, com_kdd, com_y_train, com_ktd = compress_x_train(
            kdd, y_train_full, kvd, y_val, ktd, y_test, n_com)
        # Compute accuracy
        com_mu = com_ktd @ np.linalg.inv(com_kdd) @ com_y_train
        com_acc = sum(np.argmax(y_test,axis=1) == np.argmax(com_mu,axis=1)) / len(mu)
        com_accs.append(com_acc)

   
    ns_com_dense = np.arange(1,1000)
    cost = ns_com_dense**3
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    curve1, = ax1.plot(ns_com, com_accs, 'o-', label="Accuracy", color='r')
    curve2, = ax2.plot(ns_com_dense, cost, label="Cost", color='b')
    curves = [curve1, curve2]
    ax1.legend(curves, [curve.get_label() for curve in curves])

    ax1.set(xlabel = 'Number of training sample', ylabel='Accuracy') 
    ax2.set(ylabel='Cost') 

    # print(acc, com_acc)
    # plt.figure()
    # plt.plot(ns_com, com_accs, )
    plt.title('Training set compression')
    # plt.xlabel('Number of training sample')
    # plt.ylabel('Accuracy')
    plt.show()