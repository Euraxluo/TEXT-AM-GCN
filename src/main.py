import corpus_process
import model
import graph_build_process
import pre_process
import train_process

if __name__ == "__main__":
    datasets = "logs"
    layer = model.SFGCN # model.GCN
    pre_process.PreProcess(datasets).build_features_corpus(length=2000)
    corpus_process.CorpusProcess(datasets)
    graph_build_process.GraphBuildProcess(datasets)
    if layer == model.SFGCN:
        pre_process.PreProcess(datasets).build_features_matrix()
        pre_process.PreProcess(datasets).build_knn_graph(5)
        train_process.TrainProcess(datasets, 1,layer,max_epoch=20,k=5).train()
    else:
        train_process.TrainProcess(datasets, 1,layer,max_epoch=20).train()