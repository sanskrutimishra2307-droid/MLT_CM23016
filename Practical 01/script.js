/************************************
 * 1. SYNTHETIC TRAINING DATA
 ************************************/
const trainingData = [
  { size: 500, price: 10 },
  { size: 800, price: 15 },
  { size: 1000, price: 20 },
  { size: 1200, price: 24 },
  { size: 1500, price: 30 },
  { size: 1800, price: 35 },
  { size: 2000, price: 40 }
];

// Max values for normalization
const MAX_SIZE = 2000;
const MAX_PRICE = 40;

// 🔹 ADDED: TensorFlow version
console.log("TensorFlow.js version:", tf.version.tfjs);

/************************************
 * 2. NORMALIZE DATA
 ************************************/
const xs = tf.tensor2d(
  trainingData.map(d => d.size / MAX_SIZE),
  [trainingData.length, 1]
);

const ys = tf.tensor2d(
  trainingData.map(d => d.price / MAX_PRICE),
  [trainingData.length, 1]
);

/************************************
 * 3. CREATE MODEL
 ************************************/
const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

model.compile({
  optimizer: tf.train.sgd(0.1),
  loss: "meanSquaredError"
});

let modelReady = false;

/************************************
 * 4. TRAIN MODEL
 ************************************/
(async function trainModel() {
  console.log("⏳ Training model..."); // existing (kept)

  await model.fit(xs, ys, {
    epochs: 200,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        // 🔹 ADDED: Epoch-wise loss logging (like sample)
        if (epoch % 50 === 0) {
          console.log(`Epoch ${epoch} | Loss: ${logs.loss}`);
        }
      }
    }
  });

  modelReady = true;
  console.log("✅ Model trained successfully"); // existing (kept)
  console.log("📌 Model training completed."); // 🔹 ADDED
})();

/************************************
 * 5. PREDICT PRICE
 ************************************/
function predictPrice() {

  if (!modelReady) {
    alert("Model is still training. Please wait.");
    return;
  }

  const userSize = Number(document.getElementById("sizeInput").value);

  if (!userSize || userSize <= 0) {
    alert("Enter valid house size");
    return;
  }

  // 🔹 ADDED: Log user input
  console.log("User Input (House Size):", userSize);

  // Normalize input
  const normalizedSize = userSize / MAX_SIZE;

  const predictionTensor = model.predict(
    tf.tensor2d([normalizedSize], [1, 1])
  );

  // 🔹 ADDED: Print prediction tensor (same style as sample)
  console.log("Prediction Tensor:");
  predictionTensor.print();

  // De-normalize output
  const predictedPrice =
    predictionTensor.dataSync()[0] * MAX_PRICE;

  // 🔹 ADDED: Log final predicted value
  console.log(`Predicted Price ≈ ₹${predictedPrice.toFixed(2)} Lakhs`);

  document.getElementById("result").innerText =
    `Predicted Price ≈ ₹${predictedPrice.toFixed(2)} Lakhs`;

  drawGraph(userSize, predictedPrice);
}

/************************************
 * 6. DRAW GRAPH
 ************************************/
function drawGraph(userSize, predictedPrice) {
  const ctx = document.getElementById("priceChart").getContext("2d");

  if (window.chart) window.chart.destroy();

  window.chart = new Chart(ctx, {
    type: "scatter",
    data: {
      datasets: [
        {
          label: "Training Data",
          data: trainingData.map(d => ({ x: d.size, y: d.price })),
          backgroundColor: "blue"
        },
        {
          label: "Predicted Value",
          data: [{ x: userSize, y: predictedPrice }],
          backgroundColor: "red"
        }
      ]
    },
    options: {
      scales: {
        x: {
          title: { display: true, text: "House Size (sq ft)" }
        },
        y: {
          title: { display: true, text: "Price (Lakhs)" }
        }
      }
    }
  });
}
