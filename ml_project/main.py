import yaml
import logging
import logging.config
import dataManager
import modelManager
import utils
import metrics

logger = logging.getLogger()
def train_test_pipeline_threat(trainConfig, datalinks, download_path="../downloads/", n_samples=1,
                   proxy=None, model_path="../models/test_model.model", image_path="../images/auc_test_model.png"):
    """pipeline"""
    logger.info(f"train config: {trainConfig}")
    link = datalinks["dataForTrain"]
    if download_path:
        path_train = utils.download_yadisk_link(link, proxy, download_path)
        data = dataManager.Data(datapath=path_train)
    else: # this else just for test!!
        data = dataManager.DataCreate(n_samples=n_samples)
        data = data.create()

    if trainConfig["modelParams"]:
        model = modelManager.Model(model=trainConfig["model"], params=trainConfig["modelParams"])
    else:
        model = modelManager.Model(model=trainConfig["model"])
    if trainConfig["augmentation"]:
        data.augment()
    
    if trainConfig["standart"]:
        data.std()    
    
    if trainConfig["preproccessing"]:
        for prep in trainConfig["preproccessing"]:
            rule, name = prep.split(',')
            data.add_transformed(rule, name)

    datatrain, datatest = data.train_test_split(trainConfig["split_size"])
    model.train(datatrain)
    if model_path:
        model.save(model_path)

    preds = model.predict_score(datatest)
    labels = datatest.get_label()
    if image_path:
        roc_auc, _ = metrics.plot_roc_curve(labels, preds, name=image_path)
    else:
        roc_auc, _ = metrics.plot_roc_curve(labels, preds, name=image_path)
    logger.info(f"TEST ROC AUC: {roc_auc}")
    return roc_auc, data

def setup_logger():
    with open("configs/logger_config.yml", "rt") as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)

if __name__ == "__main__":
    setup_logger()
    logger = logging.getLogger()
    with open("configs/data_links.yml", 'r') as stream:
        datalinks = yaml.safe_load(stream)
    with open("configs/proxy.yml", 'r') as stream:
        proxy = yaml.safe_load(stream)
    with open("configs/train_config_0.yml", 'r') as stream:
        trainConfig = yaml.safe_load(stream)
    print(trainConfig)
    roc_auc, data = train_test_pipeline_threat(trainConfig, datalinks, model_path="models/model_0.model",
                                                proxy=proxy, image_path="images/auc_model_0.png", download_path="downloads/")