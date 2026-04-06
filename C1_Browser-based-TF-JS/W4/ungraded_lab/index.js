let mobilenet;
let model;
const webcam = new Webcam(document.getElementById('wc'));
const dataset = new RPSDataset();
var rockSamples=0, paperSamples=0, scissorsSamples=0;
let isPredicting = false;

async function loadMobilenet() {
  const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
}

// The train function, which is right here, is setting my ys, labels to be null
// on the dataset, and then I want it to encode three labels. So that's going
// to one-hot encode because of rock, paper, and scissors one-hot encode for
// three, and here is my actual model that I'm going to be building, similar
// to what we saw in the class. My input to this is going to be the outputs of
// mobilenet, and then I'm just going to have a 100 units of dense, and then
// three units of dense activated with softmax.

async function train() {
  dataset.ys = null;
  dataset.encodeLabels(3);
  model = tf.sequential({
    layers: [
      tf.layers.flatten({inputShape: mobilenet.outputs[0].shape.slice(1)}),
      tf.layers.dense({ units: 100, activation: 'relu'}),
      tf.layers.dense({ units: 3, activation: 'softmax'})
    ]
  });

  // Then when I actually train it, I'm just going to train it with my dataset
  // xs on my input values and my dataset ys on my output values. Just typical,
  // standard training here from this, and once it's trained, I have my model.
  const optimizer = tf.train.adam(0.0001);
  model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});
  let loss = 0;
  model.fit(dataset.xs, dataset.ys, {
    epochs: 10,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        loss = logs.loss.toFixed(5);
        console.log('LOSS: ' + loss);
        }
      }
   });
}
// Let's look at the code. I can stop predicting first. So if we start looking 
// at the code, the key thing is in this handleButton function. This
// handleButton gets called whenever I click one of the buttons.
// Based on the ID of the button if it's zero, one, or two, I'm incrementing
// my number of rock samples, paper samples, or scissor samples.


//Then I'm going to be pulling frames from the web camera, passing them to the model, and getting the inference back. So take a look at this code and download this from the GitHub and have a play with it, get it working in the browser like this, either by putting it on a web server or using a local web server like I'm using, as you can see here, I'm on 127.0.0.1. Then once you've done that, you'll be ready to start looking at the exercise which is to take rock, paper, scissors and then add spock and lizards to that. 

function handleButton(elem){
	switch(elem.id){
		case "0":
			rockSamples++;
			document.getElementById("rocksamples").innerText = "Rock samples:" + rockSamples;
			break;
		case "1":
			paperSamples++;
			document.getElementById("papersamples").innerText = "Paper samples:" + paperSamples;
			break;
		case "2":
			scissorsSamples++;
			document.getElementById("scissorssamples").innerText = "Scissors samples:" + scissorsSamples;
			break;
	}
	label = parseInt(elem.id);
	const img = webcam.capture();
	// But most importantly is here, is that I'm going to add to a dataset an
    // example. So I'm going to say mobilenet.predict, the image. So it's grabbed
    // the image from the camera, and here's the actual label for it, so the
    // prediction, the label, etc., is going to get added to the dataset. That
    // dataset you can see in this JavaScript. So whenever I add an example, I'm
    // just storing all of those for later retraining. I can add my examples of
    // rock, paper, scissors. Later on, when I'm training, I'm going to be using
    // this dataset, which we will see as we go through the code.
	dataset.addExample(mobilenet.predict(img), label);

}

async function predict() {
  while (isPredicting) {
    const predictedClass = tf.tidy(() => {
      const img = webcam.capture();
      const activation = mobilenet.predict(img);
      const predictions = model.predict(activation);
      return predictions.as1D().argMax();
    });
    const classId = (await predictedClass.data())[0];
    var predictionText = "";
    switch(classId){
		case 0:
			predictionText = "I see Rock";
			break;
		case 1:
			predictionText = "I see Paper";
			break;
		case 2:
			predictionText = "I see Scissors";
			break;
	}
	document.getElementById("prediction").innerText = predictionText;
			
    
    predictedClass.dispose();
    await tf.nextFrame();
  }
}

// So my doTraining will call train.
function doTraining(){
	train();
}

function startPredicting(){
	isPredicting = true;
	predict();
}

function stopPredicting(){
	isPredicting = false;
	predict();
}

async function init(){
	await webcam.setup();
	mobilenet = await loadMobilenet();
	tf.tidy(() => mobilenet.predict(webcam.capture()));
		
}



init();


// So take a look at this code and download this from the GitHub and have a
// play with it, get it working in the browser like this, either by putting
// it on a web server or using a local web server like I'm using, as you can
// see here, I'm on 127.0.0.1. Then once you've done that, you'll be ready to
// start looking at the exercise which is to take rock, paper, scissors and
// then add spock and lizards to that.