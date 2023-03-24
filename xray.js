const fs = require('fs');
const path = require('path');
const Jimp = require('jimp');
const brain = require('brain.js');
// use dotenv to load environment variables from a .env file
require('dotenv').config();
const inputFolder = path.join(__dirname, 'chest-Xray');
const outputFolder = path.join(__dirname, 'normalized-chest-Xray');

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

    if (!stats.isFile()) {
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

  // Convertir l'image en niveaux de gris
  image.greyscale();

  // Appliquer un filtre de contraste automatique
  image.contrast(0.5);
  // redimensionner l'image
  image.resize(1000, 1000);

  // Normaliser l'intensité des pixels entre 0 et 1
  image.normalize();

  image.quality(70); 
  // to jpeg
  image.getBufferAsync(Jimp.MIME_JPEG);

  return image;
}

// normalizeImagesInFolder(path.resolve(inputFolder + '/test/NORMAL'), path.resolve(outputFolder + '/test/NORMAL')); //done
// normalizeImagesInFolder(path.resolve(inputFolder + '/test/PNEUMONIA'), path.resolve(outputFolder + '/test/PNEUMONIA')); //done
// normalizeImagesInFolder(path.resolve(inputFolder + '/train/NORMAL'), path.resolve(outputFolder + '/train/NORMAL'));  //done
// normalizeImagesInFolder(path.resolve(inputFolder + '/train/PNEUMONIA'), path.resolve(outputFolder + '/train/PNEUMONIA'));  //done
// normalizeImagesInFolder(path.resolve(inputFolder + '/val/NORMAL'), path.resolve(outputFolder + '/val/NORMAL'));  //done
// normalizeImagesInFolder(path.resolve(inputFolder + '/val/PNEUMONIA'), path.resolve(outputFolder + '/val/PNEUMONIA'));  //done

console.log('Normalisation des images en cours...');
/* PARTIE IA
*
*
*/
trainingIA();
function trainingIA(){
  // Définir la structure du réseau de neurones
  const net = new brain.NeuralNetwork({
    // Nombre de neurones dans la couche d'entrée
    // inputSize: 1000 * 1000,

    // Nombre de neurones dans la couche de sortie (2 pour une classification binaire)
    // outputSize: 2,

    // Nombre de couches cachées et nombre de neurones dans chaque couche
    hiddenLayers: [64, 32, 16],
  });

  // Charger les images normalisées et les étiquettes de classe
  const dataset = [];

  const healthyFolder = path.join(__dirname, 'normalized-chest-Xray/train/NORMAL');
  const pneumoniaFolder = path.join(__dirname, 'normalized-chest-Xray/train/PNEUMONIA');

  const healthyFiles = fs.readdirSync(healthyFolder);
  const pneumoniaFiles = fs.readdirSync(pneumoniaFolder);
  console.log(healthyFiles.length);
  console.log(pneumoniaFiles.length);
  console.log(Math.floor(healthyFiles.length / process.env.DATA_DIVISER));
  console.log(Math.floor(pneumoniaFiles.length / process.env.DATA_DIVISER));

  const NUMBEROFELEMENTS = Math.floor(healthyFiles.length / process.env.DATA_DIVISER)

  for (let i = 0; i < NUMBEROFELEMENTS; i++) {
    const file = healthyFiles[i];
    const filePath = path.join(healthyFolder, file);
    const buffer = fs.readFileSync(filePath);
    const rawimage = Jimp.decoders['image/jpeg'](buffer);
    const image = new Jimp(rawimage).greyscale();
    // to json parse
    const jsonbuffer = Array.prototype.slice.call(image.bitmap.data).map((value) => value / 255);


    dataset.push({
      input: jsonbuffer,
      output: [0], // "poumon sain"
    });
    console.log("poumon sain" + i + " / " + NUMBEROFELEMENTS);
  }
  


  for (let i = 0; i < Math.floor(NUMBEROFELEMENTS); i++) {
    const file = pneumoniaFiles[i];
    const filePath = path.join(pneumoniaFolder, file);
    const buffer = fs.readFileSync(filePath);
    const rawimage = Jimp.decoders['image/jpeg'](buffer);
    const image = new Jimp(rawimage).greyscale();
    // to json parse
    const jsonbuffer = Array.prototype.slice.call(image.bitmap.data).map((value) => value / 255);
    // 0-255 to 0-1



    dataset.push({
      input: jsonbuffer,
      output: [1], // "poumon atteint de pneumonie"
    });
    
    console.log("poumon atteint de pneumonie" + i + " / " + Math.floor(pneumoniaFiles.length / process.env.DATA_DIVISER));
  }

  // Mélanger les données d'entraînement
  const shuffledDataset = shuffleArray(dataset);





  // Entraîner le réseau de neurones
  net.train(dataset, {
    errorThresh: 0.005,
    iterations: 200,
    log: true,
    logPeriod: 1,
    learningRate: 0.3,
  });

  // Évaluer le réseau de neurones sur l'ensemble de validation
  const accuracy = brain.util.getBinaryAccuracy(net, validationData);
  console.log(`Accuracy: ${accuracy}`);
}


function shuffleArray(array) {
  let currentIndex = array.length, temporaryValue, randomIndex;

  // Tant qu'il reste des éléments à mélanger
  while (0 !== currentIndex) {

    // Choisir un élément restant au hasard
    randomIndex = Math.floor(Math.random() * currentIndex);
    currentIndex -= 1;

    // Échanger avec l'élément actuel
    temporaryValue = array[currentIndex];
    array[currentIndex] = array[randomIndex];
    array[randomIndex] = temporaryValue;
  }

  return array;
}