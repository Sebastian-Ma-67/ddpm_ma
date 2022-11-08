from DiffusionFreeGuidence.TrainCondition import train, eval,eval_gen


def main(model_config=None):
    modelConfig = {
        # "state": "train", # or eval
        "state": "eval", # or eval
        "epoch": 70,
        "batch_size": 80,
        "T": 500,
        
        "channel": 128,
        "channel_mult": [1, 2, 2, 2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.5,
        
        "beta_1": 1e-4,
        "beta_T": 0.028,
        
        "img_size": 32,
        "grad_clip": 1.,
        "device": "cuda:3",
        
        "w": 1.8,
        
        "save_dir": "./CheckpointsCondition/",
        
        "training_load_weight": None,
        "test_load_weight": "ckpt_63_.pt",
        
        "sampled_dir": "./SampledImgs/",
        "sample_gen_dir": "./test/",
        "sampledNoisyImgName": "NoisyGuidenceImgs.png",
        "sampledImgName": "SampledGuidenceImgs.png",
        "nrow": 8,
        "test_iter": 0,
        "label": 0,
        # "gen": True
        "gen": False
        
    }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
    elif modelConfig["gen"] == True:
        # for i in range (3):
        for i in range (9, 10):
            modelConfig["label"] = i
            for j in range (6):
                modelConfig["test_iter"] = j
                eval_gen(modelConfig)
    else:
        eval(modelConfig)

if __name__ == '__main__':
    main()
