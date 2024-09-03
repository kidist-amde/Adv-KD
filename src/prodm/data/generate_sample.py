from prodm.models.ddpm import MyDDPMPipeline
import os

def generate_sample(model):
    output = model()
    return output["intermediate"]

model_id = "google/ddpm-celebahq-256" 
# model = MyDDPMPipeline.from_pretrained(model_id)
# images = generate_sample(model)
# output_path = "../logs"
# for idx,img in enumerate(images):
#     img[0].save(f"../logs/image{idx}.png")

def generate_dataset(number_of_samples,model_id,output_path):
    model = MyDDPMPipeline.from_pretrained(model_id)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    start_number = 0
    while os.path.exists(os.path.join(output_path,str(start_number))):
        start_number+=1
    print("starting from",start_number)
    for i in range(start_number,number_of_samples):
        images = generate_sample(model)
        if not os.path.exists(os.path.join(output_path,str(i))):
            os.makedirs(os.path.join(output_path,str(i)))
        for idx,img in enumerate(images):
            img[0].save(os.path.join(output_path,str(i),str(idx)+".png"))
    
      
generate_dataset(20000,model_id,"../dataset")



    