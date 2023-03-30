const fs = require('fs');
const path = require('path');
const Jimp = require('jimp');
const brain = require('brain.js');
// $env:NODE_OPTIONS="--max-old-space-size=16384"
// use dotenv to load environment variables from a .env file
require('dotenv').config();
const inputFolder = path.join(__dirname, 'chest-Xray');
const outputFolder = path.join(__dirname, 'normalized-chest-Xray');
// $env:NODE_OPTIONS="--max-old-space-size=16392"

const net = new brain.NeuralNetwork({

  layers: [
    // Couche de convolution avec 8 filtres de 3x3 pixels
    {
      type: 'convolutional',
      filters: 4,
      size: 3,
      stride: 1,
      pad: 1,
      activation: 'relu',
      inputShape: [process.env.IMAGE_WIDTH, process.env.IMAGE_HEIGHT, 1],
    },
    // Couche de pooling pour réduire la taille de l'image
    {
      type: 'pooling',
      size: 2,
      stride: 2,
    },
    // Couche entièrement connectée pour la classification
    {
      type: 'dense',
      size: 10,
      activation: 'sigmoid',
    },
  ],

});




if (process.env.IMAGE_VIEW === "true") {
  viewImage();
  // wait 5 seconds*
  console.log('Création des images de visualisation en cours...');
  setTimeout(function () {
    trainingIA();
    evaluate(net);
  }, 10000);

} else {
  if (process.env.IMAGE_NORMALIZE === "false") {
    trainingIA();
    evaluate(net);
  }
}



async function normalizeImagesInFolder(inputFolder, outputFolder) {
  const files = fs.readdirSync(inputFolder);


  // Créer le dossier de sortie s'il n'existe pas
  if (!fs.existsSync(outputFolder)) {
    fs.mkdirSync(outputFolder);
  }

  for (const file of files) {

    // Ignorer les fichiers qui ne sont pas des images
    if (!file.match(/\.(jpg|jpeg|png)$/)) {
      continue;
    }

    const inputFile = path.join(inputFolder, file);
    const stats = fs.statSync(inputFile);

    if (!stats.isFile()) { // Ignorer les dossiers
      continue;
    }

    const normalizedImage = await normalizeImage(inputFile);
    const outputFile = path.join(outputFolder, file);

    await normalizedImage.writeAsync(outputFile);
  }
  console.log(`Images normalisées dans le dossier ${outputFolder}`);
}

async function normalizeImage(imagePath) {
  const image = await Jimp.read(imagePath);

  // redimensionner l'image
  image.resize(1000, 1000);
  image.greyscale();
  // to jpeg
  image.getBufferAsync(Jimp.MIME_JPEG);

  return image;
}
if (process.env.IMAGE_NORMALIZE === "true") {

  // create dir if not exist
  if (!fs.existsSync(outputFolder)) {
    fs.mkdirSync(outputFolder);
    fs.mkdirSync(outputFolder + '/test');
    fs.mkdirSync(outputFolder + '/test/NORMAL');
    fs.mkdirSync(outputFolder + '/test/PNEUMONIA');
    fs.mkdirSync(outputFolder + '/train');
    fs.mkdirSync(outputFolder + '/train/NORMAL');
    fs.mkdirSync(outputFolder + '/train/PNEUMONIA');
    fs.mkdirSync(outputFolder + '/val');
    fs.mkdirSync(outputFolder + '/val/NORMAL');
    fs.mkdirSync(outputFolder + '/val/PNEUMONIA');
    fs.mkdirSync(outputFolder + '/view/NORMAL');
    fs.mkdirSync(outputFolder + '/view/PNEUMONIA');
  }
  normalizeImagesInFolder(path.resolve(inputFolder + '/test/NORMAL'), path.resolve(outputFolder + '/test/NORMAL')); //done
  normalizeImagesInFolder(path.resolve(inputFolder + '/test/PNEUMONIA'), path.resolve(outputFolder + '/test/PNEUMONIA')); //done
  normalizeImagesInFolder(path.resolve(inputFolder + '/train/NORMAL'), path.resolve(outputFolder + '/train/NORMAL'));  //done
  normalizeImagesInFolder(path.resolve(inputFolder + '/train/PNEUMONIA'), path.resolve(outputFolder + '/train/PNEUMONIA'));  //done
  normalizeImagesInFolder(path.resolve(inputFolder + '/val/NORMAL'), path.resolve(outputFolder + '/val/NORMAL'));  //done
  normalizeImagesInFolder(path.resolve(inputFolder + '/val/PNEUMONIA'), path.resolve(outputFolder + '/val/PNEUMONIA'));  //done
}
console.log('Normalisation des images en cours...');

async function viewImage() {
  console.log('Création des images de visualisation en cours...');
  const filesNormals = fs.readdirSync(inputFolder + '/test/NORMAL')
  const filesPneumonia = fs.readdirSync(inputFolder + '/test/PNEUMONIA')

  console.log("filesNormals", filesNormals);
  console.log("filesPneumonia", filesPneumonia);

  // take the first image
  const fileNormal = filesNormals[Math.floor(Math.random() * filesNormals.length)];
  const filePneumonia = filesPneumonia[Math.floor(Math.random() * filesPneumonia.length)];
  const inputFileNormal = path.join(inputFolder + '/test/NORMAL', fileNormal);
  const inputFilePneumonia = path.join(inputFolder + '/test/PNEUMONIA', filePneumonia);

  console.log("inputFileNormal", inputFileNormal);
  console.log("inputFilePneumonia", inputFilePneumonia);

  const imageNormal = await Jimp.read(inputFileNormal)
  const imagePneumonia = await Jimp.read(inputFilePneumonia)
  console.log("modification des images en cours...")

  imageNormal.greyscale()
  imageNormal.resize(Number(process.env.IMAGE_HEIGHT), Number(process.env.IMAGE_WIDTH))
  imageNormal.threshold({ max: 150 })
  imageNormal.blur(1);

  imagePneumonia.greyscale()
  imagePneumonia.resize(Number(process.env.IMAGE_HEIGHT), Number(process.env.IMAGE_WIDTH))
  imagePneumonia.threshold({ max: 150 })
  imagePneumonia.blur(1);



  // get jpeg buffer
  const imageNormalBuffer = await imageNormal.getBufferAsync(Jimp.MIME_JPEG);
  const imagePneumoniaBuffer = await imagePneumonia.getBufferAsync(Jimp.MIME_JPEG);
  
  await fs.promises.writeFile(path.join( outputFolder + '/view/NORMAL', 'viewNormal.jpg'), imageNormalBuffer);
  await fs.promises.writeFile(path.join( outputFolder + '/view/PNEUMONIA', 'viewPneumonia.jpg'), imagePneumoniaBuffer);

  console.log('Images de visualisation enregistrées dans le dossier view');
}


function trainingIA() {

  // Charger les données d'entraînement et de validation
  const dataset = loadData(path.join(__dirname, 'normalized-chest-Xray/train/NORMAL'), path.join(__dirname, 'normalized-chest-Xray/train/PNEUMONIA'));


  // Entraîner le réseau de neurones
  net.train(dataset, {
    errorThreshold: Number(process.envIA_ERROR_THRESHOLD),
    iterations: Number(process.env.IA_ITERATIONS),
    log: (process.env.IA_LOG === 'true'),
    logPeriod: Number(process.env.IA_LOG_PERIOD),
    learningRate: Number(process.env.IA_LEARNING_RATE),
    momentum: Number(process.env.IA_MOMENTUM),
    // activation: Number(process.env.IA_ACTIVATION),
  });

  // Évaluer le réseau de neurones sur l'ensemble de validation
  const accuracy = brain.util.getBinaryAccuracy(net, validationData);
  console.log(`Accuracy: ${accuracy}`);

  // Sauvegarder le réseau de neurones
  const json = net.toJSON();
  fs.writeFileSync('xray.json', JSON.stringify(json));

}

// Évaluer le réseau de neurones sur un ensemble de données
function evaluate(net) {

  const testData = loadData(path.join(__dirname, 'normalized-chest-Xray/test/NORMAL'), path.join(__dirname, 'normalized-chest-Xray/test/PNEUMONIA'));

  let correct = 0;
  let wrong = 0;

  for (let i = 0; i < testData.length; i++) {
    const output = net.run(testData[i].input);
    console.log(`Test ${i} - Output: ${output} - Expected: ${testData[i].output}`);
    if (output === Math.round(testData[i].output)) {
      correct++;
    }
    else {
      wrong++;
    }
  }

  console.log(`Correct: ${correct} - Wrong: ${wrong}` + ` - Accuracy: ${correct / (correct + wrong)}`);

}

function loadData(healthyFolder, pneumoniaFolder, ignoreDiviser) {
  const dataset = [];

  const healthyFiles = fs.readdirSync(healthyFolder);
  const pneumoniaFiles = fs.readdirSync(pneumoniaFolder);
  console.log(healthyFiles.length);
  console.log(pneumoniaFiles.length);
  console.log(Math.floor(healthyFiles.length / process.env.DATA_DIVISER));
  console.log(Math.floor(pneumoniaFiles.length / process.env.DATA_DIVISER));

  NUMBEROFELEMENTS = healthyFiles.length;
  if (!ignoreDiviser) {
    NUMBEROFELEMENTS = Math.floor(healthyFiles.length / process.env.DATA_DIVISER);
  }

  console.log("number of elements: " + NUMBEROFELEMENTS);

  for (let i = 0; i < NUMBEROFELEMENTS; i++) {
    const file = healthyFiles[i];
    const filePath = path.join(healthyFolder, file);
    const buffer = fs.readFileSync(filePath);
    const rawimage = Jimp.decoders['image/jpeg'](buffer);
    const image = new Jimp(rawimage).greyscale().resize(Number(process.env.IMAGE_HEIGHT), Number(process.env.IMAGE_WIDTH)).threshold({ max: 150 }).blur(1);

    // write first image to file view /normalizes-chest-Xray/view/normal/0.jpg

    // to json parse
    const jsonbuffer = Array.prototype.slice.call(image.bitmap.data).map((value) => value / 255); // 0-255 to 0-1
    const jsonNormal = [];
    // use 1 data of 4 (4 = 1 pixel)
    for (let i = 0; i < jsonbuffer.length; i++) {
      const pixel = jsonbuffer.slice(i, i + 4);
      jsonNormal.push(pixel[0]);
    }

    dataset.push({
      input: jsonNormal,
      output: [0], // "poumon sain"
    });


    console.log("poumon sain : " + i + " / " + NUMBEROFELEMENTS);
  }

  for (let i = 0; i < NUMBEROFELEMENTS; i++) {
    const file = pneumoniaFiles[i];
    const filePath = path.join(pneumoniaFolder, file);
    const buffer = fs.readFileSync(filePath);
    const rawimage = Jimp.decoders['image/jpeg'](buffer);
    const image = new Jimp(rawimage).greyscale().resize(Number(process.env.IMAGE_HEIGHT), Number(process.env.IMAGE_WIDTH)).threshold({max: 150}).blur(1);
    
    // to json parse
    const jsonbuffer = Array.prototype.slice.call(image.bitmap.data).map((value) => value / 255); // 0-255 to 0-1
    const jsonPneumonia = [];

    // use 1 data of 4 (4 = 1 pixel) to keep only the greyscale and ignore alpha/color 
    for (let i = 0; i < jsonbuffer.length; i++) {
      const pixel = jsonbuffer.slice(i, i + 4);

      jsonPneumonia.push(pixel[0]);
    }

    dataset.push({
      input: jsonPneumonia,
      output: [1], // "pneumonie"
    });



    console.log("poumon atteint de pneumonie" + i + " / " + NUMBEROFELEMENTS);
  }

  console.log("dataset loaded");

  return shuttlleArray(dataset);
}



function shuttlleArray(array) {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }

  console.log("dataset shuffled");

  return array;

}