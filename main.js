import * as tf from "@tensorflow/tfjs";
async function loadDenseNetModel() {
  document.getElementById("predictionResult").innerText = `Loading Model ...
  (can take up to 1 min)`;

  // Load the DenseNet model from the local files
  // const model = await tf.loadLayersModel("/mymodel2/model.json");
  const model = await tf.loadLayersModel(
    "https://raw.githubusercontent.com/SpectralGT/ProjectK_model/refs/heads/main/model.json"
  );

  console.log("DenseNet model loaded successfully");
  document.getElementById("predictionResult").innerText = `Model Loaded
  (first prediction takes time)`;
  return model;
}

function preprocessImage(image) {
  // Convert the image to a tensor and resize it to 224x224 pixels
  let tensor = tf.browser
    .fromPixels(image)
    .resizeBilinear([224, 224]) // Resize the image to 224x224
    .toFloat()
    .expandDims(); // Add a batch dimension (shape: [1, 224, 224, 3])

  // Normalize the image to range [-1, 1]
  return tensor.div(127.5).sub(1);
}

async function classifyImage(model, imageElement) {
  const processedImage = preprocessImage(imageElement);
  const prediction = model.predict(processedImage);

  // Get the class with the highest score
  const predictedClass = prediction.argMax(-1);
  const predictedClassIndex = (await predictedClass.data())[0];
  console.log(await prediction.data());
  // Display the prediction

  const diseases = [
    "Bacterial Pneumonia",
    "Corona Virus Disease",
    "Normal",
    "Tuberculosis",
    "Viral Pneumonia",
  ];

  document.getElementById(
    "predictionResult"
  ).innerText = `Predicted Disease: ${diseases[predictedClassIndex]}`;
}

// Load the model when the page loads
let densenetModel;
window.onload = async function () {
  densenetModel = await loadDenseNetModel();

  // Set up event listener for image input
  const imageInput = document.getElementById("imageInput");
  const selectedImage = document.getElementById("selectedImage");

  imageInput.addEventListener("change", (event) => {
    const file = event.target.files[0];
    if (file) {
      // Display the selected image
      const reader = new FileReader();
      reader.onload = function (e) {
        selectedImage.src = e.target.result;
      };
      reader.readAsDataURL(file);

      // Classify the image after it's loaded
      selectedImage.onload = function () {
        classifyImage(densenetModel, selectedImage);
      };
    }
  });
};
