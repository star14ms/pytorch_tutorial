### 데이터로더는 클래스임

def make_dataloader(x, t, batch_size):
    dataloader = []

    for i in range(len(t) // batch_size):
        batch_x = x[i*batch_size : (i+1)*batch_size]
        batch_t = t[i*batch_size : (i+1)*batch_size]
        dataloader.append( (batch_x, batch_t, ) )

    return dataloader

# train_dataloader = make_dataloader(x_train, t_train, batch_size)
# test_dataloader = make_dataloader(x_test, t_test, batch_size)
