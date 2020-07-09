# Python Project on Traffic Signs Recognition with 95% Accuracy using CNN & Keras

You must have heard about the self-driving cars in which the passenger can fully depend on the car for traveling. But to achieve level 5 autonomous, it is necessary for vehicles to understand and follow all traffic rules.

In the world of Artificial Intelligence and advancement in technologies, many researchers and big companies like Tesla, Uber, Google, Mercedes-Benz, Toyota, Ford, Audi, etc are working on autonomous vehicles and self-driving cars. So, for achieving accuracy in this technology, the vehicles should be able to interpret traffic signs and make decisions accordingly.


# What is Traffic Signs Recognition?

There are several different types of traffic signs like speed limits, no entry, traffic signals, turn left or right, children crossing, no passing of heavy vehicles, etc. Traffic signs classification is the process of identifying which class a traffic sign belongs to.

# Traffic Signs Recognition – About the Python Project

In this Python project example, we will build a deep neural network model that can classify traffic signs present in the image into different categories. With this model, we are able to read and understand traffic signs which are a very important task for all autonomous vehicles.

<img style="-webkit-user-select: none;margin: auto;" src="https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/traffic-sign-recognition-project.gif" width="718" height="403">

## The Dataset of Python Project

For this project, we are using the public dataset available at Kaggle:

<p><a href="https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign" onclick="javascript:window.open('https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign'); return false;">Traffic Signs Dataset</a></p>

The dataset contains more than 50,000 images of different traffic signs. It is further classified into 43 different classes. The dataset is quite varying, some of the classes have many images while some classes have few images. The size of the dataset is around 300 MB. The dataset has a train folder which contains images inside each class and a test folder which we will use for testing our model.

## Prerequisites

This project requires prior knowledge of Keras, Matplotlib, Scikit-learn, Pandas, PIL and image classification.

To install the necessary packages used for this Python data science project, enter the below command in your terminal:

     * $ pip install tensorflow
     * $ pip install keras
     * $ pip install sklearn
     * $ pip install matplotlib
     * $ pip install pil
     
 # Steps to Build the Python Project
 
 <p>To get started with the project, download and unzip the file from this link – <a href="https://drive.google.com/open?id=1BGDHe6qQwrBEgnl-tXTSKo6TvDj8U3wS" onclick="javascript:window.open('https://drive.google.com/open?id=1BGDHe6qQwrBEgnl-tXTSKo6TvDj8U3wS'); return false;">Traffic Signs Recognition Zip File</a></p>
 
 And extract the files into a folder such that you will have a train, test and a meta folder.
 
 <img style="-webkit-user-select: none;margin: auto;cursor: zoom-in;" src="https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/dataset.png" width="718" height="252">
 
 Create a Python script file and name it traffic_signs.py in the project folder.

Our approach to building this traffic sign classification model is discussed in four steps:

<ul><li>Explore the dataset</li><li>Build a CNN model</li><li>Train and validate the model</li><li>Test the model with test dataset</li></ul>

## Step 1: Explore the dataset

Our ‘train’ folder contains 43 folders each representing a different class. The range of the folder is from 0 to 42. With the help of the OS module, we iterate over all the classes and append images and their respective labels in the data and labels list.

The PIL library is used to open image content into an array.<p><a href="https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/exploring-dataset-python-project.png" onclick="javascript:window.open('https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/exploring-dataset-python-project.png'); return false;" class=""><img class="aligncenter size-full wp-image-73400 tc-smart-load-skip tc-smart-loaded" src="https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/exploring-dataset-python-project.png" alt="exploring dataset in python project" width="992" height="657" sizes="(max-width: 992px) 100vw, 992px" srcset="https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/exploring-dataset-python-project.png 992w, https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/exploring-dataset-python-project-150x99.png 150w, https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/exploring-dataset-python-project-300x199.png 300w, https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/exploring-dataset-python-project-768x509.png 768w, https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/exploring-dataset-python-project-520x344.png 520w" style="display: block;"></a></p>

Finally, we have stored all the images and their labels into lists (data and labels).

We need to convert the list into numpy arrays for feeding to the model.

The shape of data is (39209, 30, 30, 3) which means that there are 39,209 images of size 30×30 pixels and the last 3 means the data contains colored images (RGB value).

With the sklearn package, we use the train_test_split() method to split training and testing data.

From the keras.utils package, we use to_categorical method to convert the labels present in y_train and t_test into one-hot encoding.

<p><a href="https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/splitting-dataset-python-project.png" onclick="javascript:window.open('https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/splitting-dataset-python-project.png'); return false;" class=""><img class="aligncenter size-full wp-image-73401 tc-smart-load-skip tc-smart-loaded" src="https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/splitting-dataset-python-project.png" alt="splitting dataset in python project" width="992" height="257" sizes="(max-width: 992px) 100vw, 992px" srcset="https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/splitting-dataset-python-project.png 992w, https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/splitting-dataset-python-project-150x39.png 150w, https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/splitting-dataset-python-project-300x78.png 300w, https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/splitting-dataset-python-project-768x199.png 768w, https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/splitting-dataset-python-project-520x135.png 520w" style="display: block;"></a></p>

## Step 2:Build a CNN Model

<p>To classify the images into their respective categories, we will build a CNN model (<em><strong><a href="https://data-flair.training/blogs/convolutional-neural-networks-tutorial/">Convolutional Neural Network</a></strong></em>). CNN is best for image classification purposes.</p>

The architecture of our model is:

<ul><li>2 Conv2D layer (filter=32, kernel_size=(5,5), activation=”relu”)</li><li>MaxPool2D layer ( pool_size=(2,2))</li><li>Dropout layer (rate=0.25)</li><li>2 Conv2D layer (filter=64, kernel_size=(3,3), activation=”relu”)</li><li>MaxPool2D layer ( pool_size=(2,2))</li><li>Dropout layer (rate=0.25)</li><li>Flatten layer to squeeze the layers into 1 dimension</li><li>Dense Fully connected layer (256 nodes, activation=”relu”)</li><li>Dropout layer (rate=0.5)</li><li>Dense layer (43 nodes, activation=”softmax”)</li></ul>

We compile the model with Adam optimizer which performs well and loss is “categorical_crossentropy” because we have multiple classes to categorise.

<p><a href="https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/cnn-model-python-data-science-project.png" onclick="javascript:window.open('https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/cnn-model-python-data-science-project.png'); return false;" class=""><img class="aligncenter size-full wp-image-73403 tc-smart-load-skip tc-smart-loaded" src="https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/cnn-model-python-data-science-project.png" alt="cnn model in python data science project" width="992" height="350" sizes="(max-width: 992px) 100vw, 992px" srcset="https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/cnn-model-python-data-science-project.png 992w, https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/cnn-model-python-data-science-project-150x53.png 150w, https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/cnn-model-python-data-science-project-300x106.png 300w, https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/cnn-model-python-data-science-project-768x271.png 768w, https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/cnn-model-python-data-science-project-520x183.png 520w" style="display: block;"></a></p>

## Steps 3: Train and validate the Model

After building the model architecture, we then train the model using model.fit(). I tried with batch size 32 and 64. Our model performed better with 64 batch size. And after 15 epochs the accuracy was stable.

<p><a href="https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/training-project-in-python.png" onclick="javascript:window.open('https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/training-project-in-python.png'); return false;" class=""><img class="aligncenter size-full wp-image-73404 tc-smart-load-skip tc-smart-loaded" src="https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/training-project-in-python.png" alt="training the model in project in python" width="992" height="981" sizes="(max-width: 992px) 100vw, 992px" srcset="https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/training-project-in-python.png 992w, https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/training-project-in-python-150x148.png 150w, https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/training-project-in-python-300x297.png 300w, https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/training-project-in-python-768x759.png 768w, https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/training-project-in-python-520x514.png 520w" style="display: block;"></a></p>

Our model got a 95% accuracy on the training dataset. With matplotlib, we plot the graph for accuracy and the loss.

<div id="attachment_73405" style="width: 1002px" class="wp-caption aligncenter"><a href="https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/plotting-accuracy-python-project-example.png" onclick="javascript:window.open('https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/plotting-accuracy-python-project-example.png'); return false;" class=""><img aria-describedby="caption-attachment-73405" class="wp-image-73405 size-full tc-smart-load-skip tc-smart-loaded" src="https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/plotting-accuracy-python-project-example.png" alt="plotting accuracy in python project example" width="992" height="349" sizes="(max-width: 992px) 100vw, 992px" srcset="https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/plotting-accuracy-python-project-example.png 992w, https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/plotting-accuracy-python-project-example-150x53.png 150w, https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/plotting-accuracy-python-project-example-300x106.png 300w, https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/plotting-accuracy-python-project-example-768x270.png 768w, https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/plotting-accuracy-python-project-example-520x183.png 520w" style="display: inline;"></a><p id="caption-attachment-73405" class="wp-caption-text">Plotting Accuracy</p></div>

<div id="attachment_73406" style="width: 1002px" class="wp-caption aligncenter"><a href="https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/accuracy-loss-python-machine-learning-project.png" onclick="javascript:window.open('https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/accuracy-loss-python-machine-learning-project.png'); return false;" class=""><img aria-describedby="caption-attachment-73406" class="wp-image-73406 size-full tc-smart-load-skip tc-smart-loaded" src="https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/accuracy-loss-python-machine-learning-project.png" alt="accuracy &amp; loss in python machine learning project" width="992" height="601" sizes="(max-width: 992px) 100vw, 992px" srcset="https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/accuracy-loss-python-machine-learning-project.png 992w, https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/accuracy-loss-python-machine-learning-project-150x91.png 150w, https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/accuracy-loss-python-machine-learning-project-300x182.png 300w, https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/accuracy-loss-python-machine-learning-project-768x465.png 768w, https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/accuracy-loss-python-machine-learning-project-520x315.png 520w" style="display: inline;"></a><p id="caption-attachment-73406" class="wp-caption-text">Accuracy and Loss Graphs</p></div>

## Step 4: Test our model with test dataset

Our dataset contains a test folder and in a test.csv file, we have the details related to the image path and their respective class labels. We extract the image path and labels using pandas. Then to predict the model, we have to resize our images to 30×30 pixels and make a numpy array containing all image data. From the sklearn.metrics, we imported the accuracy_score and observed how our model predicted the actual labels. We achieved a 95% accuracy in this model.

<p><a href="https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/testing-accuracy-advanced-python-project.png" onclick="javascript:window.open('https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/testing-accuracy-advanced-python-project.png'); return false;" class=""><img class="aligncenter size-full wp-image-73407 tc-smart-load-skip tc-smart-loaded" src="https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/testing-accuracy-advanced-python-project.png" alt="testing accuracy in advanced python project" width="1015" height="471" sizes="(max-width: 1015px) 100vw, 1015px" srcset="https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/testing-accuracy-advanced-python-project.png 1015w, https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/testing-accuracy-advanced-python-project-150x70.png 150w, https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/testing-accuracy-advanced-python-project-300x139.png 300w, https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/testing-accuracy-advanced-python-project-768x356.png 768w, https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/testing-accuracy-advanced-python-project-520x241.png 520w" style="display: block;"></a></p>

In the end, we are going to save the model that we have trained using the Keras model.save() function.

<div class="EnlighterJSWrapper eclipseEnlighterJSWrapper"><ol class="hoverEnabled eclipseEnlighterJS EnlighterJS"><li class=" odd"><span class="">model.</span><span class="kw1">save</span><span class="br0">(</span><span class="">‘traffic_classifier.h5’</span><span class="br0">)</span></li></ol><pre style="display: none;">model.save(‘traffic_classifier.h5’)</pre></div>

## Full Source code:
    import numpy as np 
    import pandas as pd 
    import matplotlib.pyplot as plt
    import cv2
    import tensorflow as tf
    from PIL import Image
    import os
    from sklearn.model_selection import train_test_split
    from keras.utils import to_categorical
    from keras.models import Sequential, load_model
    from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

    data = []
    labels = []
    classes = 43
    cur_path = os.getcwd()

    #Retrieving the images and their labels 
    for i in range(classes):
        path = os.path.join(cur_path,'train',str(i))
        images = os.listdir(path)

        for a in images:
            try:
                image = Image.open(path + '\\'+ a)
                image = image.resize((30,30))
                image = np.array(image)
                #sim = Image.fromarray(image)
                data.append(image)
                labels.append(i)
            except:
                print("Error loading image")

    #Converting lists into numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    print(data.shape, labels.shape)
    #Splitting training and testing dataset
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    #Converting the labels into one hot encoding
    y_train = to_categorical(y_train, 43)
    y_test = to_categorical(y_test, 43)

    #Building the model
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(43, activation='softmax'))

    #Compilation of the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    epochs = 15
    history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))
    model.save("my_model.h5")

    #plotting graphs for accuracy 
    plt.figure(0)
    plt.plot(history.history['accuracy'], label='training accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.title('Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()

    plt.figure(1)
    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    #testing accuracy on test dataset
    from sklearn.metrics import accuracy_score

    y_test = pd.read_csv('Test.csv')

    labels = y_test["ClassId"].values
    imgs = y_test["Path"].values

    data=[]

    for img in imgs:
        image = Image.open(img)
        image = image.resize((30,30))
        data.append(np.array(image))

    X_test=np.array(data)

    pred = model.predict_classes(X_test)

    #Accuracy with the test data
    from sklearn.metrics import accuracy_score
    print(accuracy_score(labels, pred))

    model.save(‘traffic_classifier.h5’)</pre></div>


# Traffic Signs Classifier GUI

Now we are going to build a graphical user interface for our traffic signs classifier with Tkinter. Tkinter is a GUI toolkit in the standard python library. Make a new file in the project folder and copy the below code. Save it as gui.py and you can run the code by typing python gui.py in the command line.

In this file, we have first loaded the trained model ‘traffic_classifier.h5’ using Keras. And then we build the GUI for uploading the image and a button is used to classify which calls the classify() function. The classify() function is converting the image into the dimension of shape (1, 30, 30, 3). This is because to predict the traffic sign we have to provide the same dimension we have used when building the model. Then we predict the class, the model.predict_classes(image) returns us a number between (0-42) which represents the class it belongs to. We use the dictionary to get the information about the class. Here’s the code for the gui.py file.

## Code:

    import tkinter as tk
    from tkinter import filedialog
    from tkinter import *
    from PIL import ImageTk, Image

    import numpy
    #load the trained model to classify sign
    from keras.models import load_model
    model = load_model('traffic_classifier.h5')

    #dictionary to label all traffic signs class.
    classes = { 1:'Speed limit (20km/h)',
                2:'Speed limit (30km/h)', 
                3:'Speed limit (50km/h)', 
                4:'Speed limit (60km/h)', 
                5:'Speed limit (70km/h)', 
                6:'Speed limit (80km/h)', 
                7:'End of speed limit (80km/h)', 
                8:'Speed limit (100km/h)', 
                9:'Speed limit (120km/h)', 
                10:'No passing', 
                11:'No passing veh over 3.5 tons', 
                12:'Right-of-way at intersection', 
                13:'Priority road', 
                14:'Yield', 
                15:'Stop', 
                16:'No vehicles', 
                17:'Veh &gt; 3.5 tons prohibited', 
                18:'No entry', 
                19:'General caution', 
                20:'Dangerous curve left', 
                21:'Dangerous curve right', 
                22:'Double curve', 
                23:'Bumpy road', 
                24:'Slippery road', 
                25:'Road narrows on the right', 
                26:'Road work', 
                27:'Traffic signals', 
                28:'Pedestrians', 
                29:'Children crossing', 
                30:'Bicycles crossing', 
                31:'Beware of ice/snow',
                32:'Wild animals crossing', 
                33:'End speed + passing limits', 
                34:'Turn right ahead', 
                35:'Turn left ahead', 
                36:'Ahead only', 
                37:'Go straight or right', 
                38:'Go straight or left', 
                39:'Keep right', 
                40:'Keep left', 
                41:'Roundabout mandatory', 
                42:'End of no passing', 
                43:'End no passing veh &gt; 3.5 tons' }

    #initialise GUI
    top=tk.Tk()
    top.geometry('800x600')
    top.title('Traffic sign classification')
    top.configure(background='#CDCDCD')

    label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
    sign_image = Label(top)

    def classify(file_path):
        global label_packed
        image = Image.open(file_path)
        image = image.resize((30,30))
        image = numpy.expand_dims(image, axis=0)
        image = numpy.array(image)
        pred = model.predict_classes([image])[0]
        sign = classes[pred+1]
        print(sign)
        label.configure(foreground='#011638', text=sign) 

    def show_classify_button(file_path):
        classify_b=Button(top,text="Classify Image",command=lambda: classify(file_path),padx=10,pady=5)
        classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
        classify_b.place(relx=0.79,rely=0.46)

    def upload_image():
        try:
            file_path=filedialog.askopenfilename()
            uploaded=Image.open(file_path)
            uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
            im=ImageTk.PhotoImage(uploaded)

            sign_image.configure(image=im)
            sign_image.image=im
            label.configure(text='')
            show_classify_button(file_path)
        except:
            pass

    upload=Button(top,text="Upload an image",command=upload_image,padx=10,pady=5)
    upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))

    upload.pack(side=BOTTOM,pady=50)
    sign_image.pack(side=BOTTOM,expand=True)
    label.pack(side=BOTTOM,expand=True)
    heading = Label(top, text="Know Your Traffic Sign",pady=20, font=('arial',20,'bold'))
    heading.configure(background='#CDCDCD',foreground='#364156')
    heading.pack()
    top.mainloop()


## Output:

<p><a href="https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/graphical-user-interface-project-in-python.png" onclick="javascript:window.open('https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/graphical-user-interface-project-in-python.png'); return false;" class=""><img class="aligncenter size-full wp-image-73408 tc-smart-load-skip tc-smart-loaded" src="https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/graphical-user-interface-project-in-python.png" alt="graphical user interface for project in python" width="678" height="511" sizes="(max-width: 678px) 100vw, 678px" srcset="https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/graphical-user-interface-project-in-python.png 678w, https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/graphical-user-interface-project-in-python-150x113.png 150w, https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/graphical-user-interface-project-in-python-300x226.png 300w, https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/12/graphical-user-interface-project-in-python-520x392.png 520w" style="display: block;"></a></p>

## Summary

In this Python project with source code, we have successfully classified the traffic signs classifier with 95% accuracy and also visualized how our accuracy and loss changes with time, which is pretty good from a simple CNN model.

Follow Me On Instagram at <a href = "https://www.instagram.com/_hemanth_shetty__/">@_hemanth_shetty__</a>

#### ThankYou!
