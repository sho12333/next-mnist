"use client";

import React, { useEffect, useRef, useState } from "react";
import { InferenceSession, Tensor } from "onnxruntime-web";

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [context, setContext] = useState<CanvasRenderingContext2D | null>(null);
  const [isDrawing, setIsDrawing] = useState<Boolean>(false);
  const [predictions, setPredictions] = useState<number>(0);

  useEffect(() => {
    if (canvasRef.current) {
      const canvas = canvasRef.current;
      const context = canvas.getContext("2d");
      if (context) {
        context.strokeStyle = "red";
        context.lineCap = "round";
        context.lineWidth = 5;
        setContext(context);
      }
    }
  }, [canvasRef, context]);

  const startDrawing = ({ nativeEvent }: React.MouseEvent) => {
    const { offsetX, offsetY } = nativeEvent;
    if (context) {
      context.beginPath();
      context.moveTo(offsetX, offsetY);
    }
    setIsDrawing(true);
  };

  const draw = ({ nativeEvent }: React.MouseEvent) => {
    if (!isDrawing) return;

    const { offsetX, offsetY } = nativeEvent;
    if (context) {
      context.lineTo(offsetX, offsetY);
      context.stroke();
    }
  };

  const finishDrawing = () => {
    setIsDrawing(false);
    if (context) {
      context.closePath();
    }
  };

  const clearCanvas = () => {
    if (context) {
      context.clearRect(0, 0, context.canvas.width, context.canvas.height);
    }
  };

  async function loadModel(): Promise<InferenceSession> {
    const session = await InferenceSession.create("/models/onnx_model.onnx", {
      executionProviders: ["webgl"],
      graphOptimizationLevel: "all",
    });
    return session;
  }

  async function updatePredictions() {
    if (!canvasRef.current) return;
    if (!context) return;
    const imgData = context.getImageData(
      0,
      0,
      canvasRef.current.width,
      canvasRef.current.height
    );

    var normalArray = Array.from(imgData.data);
    const inputData = new Tensor("float32", normalArray);
    const session = await loadModel();
    const feeds = { "0": inputData };
    const output = await session.run(feeds);
    const predictions = Object.values(output)[0];
    const predictionData = predictions.data as Float32Array;
    var resultArray = Array.from(predictionData);
    const maxPredictionIndex = resultArray.indexOf(Math.max(...resultArray));
    setPredictions(maxPredictionIndex);
  }

  return (
    <main>
      <div className="flex flex-col items-center space-y-4">
        <h1 className="text-4xl font-bold mt-6">MNIST</h1>
        <p className="text-xl font-bold">Handwritten Digit Recognition</p>

        <canvas
          id="canvas"
          ref={canvasRef}
          onMouseDown={startDrawing}
          onMouseMove={draw}
          onMouseUp={finishDrawing}
          onMouseLeave={finishDrawing}
          width={280}
          height={280}
          className="border-2 border-black"
        />
        <div className="flex space-x-4">
          <button
            onClick={clearCanvas}
            className="bg-black text-white px-4 py-2 rounded-md"
          >
            Clear
          </button>
          <button
            onClick={updatePredictions}
            className="bg-black text-white px-4 py-2 rounded-md"
          >
            Predict
          </button>
        </div>
        <p className="text-2xl font-bold mt-6">Prediction: {predictions}</p>
      </div>
    </main>
  );
}
