import tensorflow as tf

class Model:
  def __init__(self,model,result_target:float,preprocess_layer):
    self.model=model
    self.model_name=model.name
    self.result_target=result_target
    self.type_result=self.get_type_result()

    self.model_predicts_the_correct_class=None
    self.accuracy_in_the_class_of_this_image=None
    self.threshold_target_accuracy_in_the_class_of_this_image=None
    self.is_necessary_continue_modified_because_the_image_target_not_achieve=None
    # self.calculate_loss=True

    self.preprocess_layer=preprocess_layer
    
  def get_type_result(self):
    if self.result_target==1:
      return "improving model 100% accuracy"
    elif self.result_target==0:
      return "The model always fail"
    else:
      return "The model is random with a accuracy of {} % ".format(self.result_target*100)
    
  def is_next_iteracion_model_predicts_the_correct_class(self):
      random_number=tf.random.uniform(
            [], minval=0.000001, maxval=1, dtype=tf.dtypes.float32, seed=None, name=None
            )
      if self.result_target-random_number>=0: 

        self.model_predicts_the_correct_class=True
        self.accuracy_in_the_class_of_this_image=0
        self.threshold_target_accuracy_in_the_class_of_this_image=0.95

      else:

        self.model_predicts_the_correct_class=False
        self.accuracy_in_the_class_of_this_image=1
        self.threshold_target_accuracy_in_the_class_of_this_image=0.01

  def predict(self,image):
    preprocess_image=self.preprocess_layer(image)
    self.prediction=self.model(preprocess_image,training=False)

  def update_accuracy_in_the_class_of_this_image(self,target_class):
    self.accuracy_in_the_class_of_this_image=self.prediction[0][target_class].numpy()
  # def continue_calculate_loss(self):

  #   if self.model_predicts_the_correct_class and self.accuracy_in_the_class_of_this_image<self.threshold_target_accuracy_in_the_class_of_this_image:
  #       self.calculate_loss=True

  #   elif  not self.model_predicts_the_correct_class and self.accuracy_in_the_class_of_this_image>self.threshold_target_accuracy_in_the_class_of_this_image:
  #       self.calculate_loss=True

  #   else:
  #       self.calculate_loss=False