from Evaluation import evaluation
from Models import resigmoidel
from TrainingAndValidation import train_test
import augmentation
import preprocessing
import result

#preprocessing_section
images_montgomery,labels_montgomery,images_china,labels_china,num_normal_montgomery,num_abnormal_montgomery,num_normal_china,num_abnormal_china = preprocessing.process()
#augmentation
images_montgomery,labels_montgomery,images_china,labels_china,augmented_images_montgomery,augmented_labels_montgomery = augmentation.process(images_montgomery,labels_montgomery,images_china,labels_china,num_normal_montgomery,num_abnormal_montgomery,num_normal_china,num_abnormal_china)

#data preparing for model
X_train, X_test, y_train, y_test = train_test.train_test(images_montgomery, 
                                                         images_china, 
                                                         augmented_images_montgomery,
                                                         labels_montgomery, 
                                                         labels_china,
                                                        augmented_labels_montgomery)

        #for resnet
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)

#evaluation
model = evaluation.eva(X_train, X_test, y_train, y_test)

#plots and results
result.res(model,X_train, X_test, y_train, y_test)
