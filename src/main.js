import {
  HandLandmarker,
  FaceLandmarker,
  FilesetResolver
} from "@mediapipe/tasks-vision";
import './style.css';

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("fog-canvas");
const canvasCtx = canvasElement.getContext("2d");
const loadingScreen = document.getElementById("loading");
const instructionsScreen = document.getElementById("instructions");

let handLandmarker;
let faceLandmarker;
let lastVideoTime = -1;

// Base fog level
const MAX_FOG = 0.85; 

// Store previous coordinates for smooth wiping
const lastPositions = new Map();

async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: 1280,
      height: 720,
      facingMode: "user"
    },
    audio: false,
  });
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

async function initializeMediaPipe() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.12/wasm"
  );
  
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numHands: 2
  });

  faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    outputFaceBlendshapes: true,
    numFaces: 1
  });
}

function resizeCanvas() {
  canvasElement.width = video.videoWidth;
  canvasElement.height = video.videoHeight;
  
  // Fill initial fog
  canvasCtx.fillStyle = `rgba(255, 255, 255, ${MAX_FOG})`;
  canvasCtx.fillRect(0, 0, canvasElement.width, canvasElement.height);
}

function drawSoftWipeCircle(x, y, radius) {
  // Use a slightly larger solid core (0.4) for a cleaner wipe
  const gradient = canvasCtx.createRadialGradient(x, y, radius * 0.4, x, y, radius);
  gradient.addColorStop(0, 'rgba(0, 0, 0, 1)');
  gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
  
  canvasCtx.fillStyle = gradient;
  canvasCtx.beginPath();
  canvasCtx.arc(x, y, radius, 0, 2 * Math.PI);
  canvasCtx.fill();
}

function wipeFog(id, x, y, radius) {
  // Mirror X coordinate because video is mirrored via CSS scaleX(-1)
  const mappedX = (1 - x) * canvasElement.width;
  const mappedY = y * canvasElement.height;
  
  canvasCtx.globalCompositeOperation = 'destination-out';
  
  const lastPos = lastPositions.get(id);
  
  if (lastPos) {
    // Interpolate points between last position and current position for a smooth wipe
    const dx = mappedX - lastPos.x;
    const dy = mappedY - lastPos.y;
    const dist = Math.sqrt(dx * dx + dy * dy);
    const steps = Math.max(1, Math.ceil(dist / (radius * 0.3))); // Draw circle every 0.3x radius
    
    for (let i = 0; i <= steps; i++) {
      const interpX = lastPos.x + (dx * (i / steps));
      const interpY = lastPos.y + (dy * (i / steps));
      drawSoftWipeCircle(interpX, interpY, radius);
    }
  } else {
    drawSoftWipeCircle(mappedX, mappedY, radius);
  }
  
  lastPositions.set(id, { x: mappedX, y: mappedY });
}

function addBreathFog(x, y, intensity) {
  const mappedX = (1 - x) * canvasElement.width;
  const mappedY = y * canvasElement.height;
  
  canvasCtx.globalCompositeOperation = 'source-over';
  
  // Increased radius significantly to fill a larger area
  const radius = 800 * intensity; 
  const gradient = canvasCtx.createRadialGradient(mappedX, mappedY, 0, mappedX, mappedY, radius);
  
  // Make the fog appear much faster and denser
  gradient.addColorStop(0, `rgba(255, 255, 255, ${0.8 * intensity})`);
  gradient.addColorStop(0.5, `rgba(255, 255, 255, ${0.3 * intensity})`);
  gradient.addColorStop(1, 'rgba(255, 255, 255, 0)');
  
  canvasCtx.fillStyle = gradient;
  canvasCtx.beginPath();
  canvasCtx.arc(mappedX, mappedY, radius, 0, 2 * Math.PI);
  canvasCtx.fill();
}

function distance(p1, p2) {
  const dx = p1.x - p2.x;
  const dy = p1.y - p2.y;
  return Math.sqrt(dx * dx + dy * dy);
}

function isHandOpen(landmarks) {
  const wrist = landmarks[0];
  const middleExtended = distance(wrist, landmarks[12]) > distance(wrist, landmarks[10]);
  const ringExtended = distance(wrist, landmarks[16]) > distance(wrist, landmarks[14]);
  const pinkyExtended = distance(wrist, landmarks[20]) > distance(wrist, landmarks[18]);
  
  // If at least 2 of the 3 other fingers (middle, ring, pinky) are extended, we consider it an open hand
  return (middleExtended ? 1 : 0) + (ringExtended ? 1 : 0) + (pinkyExtended ? 1 : 0) >= 2;
}

async function renderLoop() {
  let startTimeMs = performance.now();
  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    
    // Process Hands
    if (handLandmarker) {
      const handResults = handLandmarker.detectForVideo(video, startTimeMs);
      
      const activeIds = new Set();

      if (handResults.landmarks && handResults.landmarks.length > 0) {
        handResults.landmarks.forEach((landmarks, handIndex) => {
          
          if (isHandOpen(landmarks)) {
            // OPEN HAND: Wipe a large area
            const pointsToWipe = [
              { id: `thumb-${handIndex}`, lm: landmarks[4], radius: 60 },
              { id: `index-${handIndex}`, lm: landmarks[8], radius: 60 },
              { id: `middle-${handIndex}`, lm: landmarks[12], radius: 60 },
              { id: `ring-${handIndex}`, lm: landmarks[16], radius: 60 },
              { id: `pinky-${handIndex}`, lm: landmarks[20], radius: 60 },
              // Palm base, center, and top to cover the whole hand
              { id: `wrist-${handIndex}`, lm: landmarks[0], radius: 100 },
              { id: `palm-center-${handIndex}`, lm: landmarks[9], radius: 110 }
            ];

            pointsToWipe.forEach(pt => {
              activeIds.add(pt.id);
              wipeFog(pt.id, pt.lm.x, pt.lm.y, pt.radius);
            });
          } else {
            // POINTING: Wipe ONLY with the index finger tip, like a pen
            const indexTip = landmarks[8];
            const indexId = `index-${handIndex}`;
            
            activeIds.add(indexId);
            // Small radius to simulate drawing with a single finger
            wipeFog(indexId, indexTip.x, indexTip.y, 35);
          }
        });
      }
      
      // Cleanup old positions so strokes don't connect randomly if hand disappears
      for (const id of lastPositions.keys()) {
        if (!activeIds.has(id)) {
          lastPositions.delete(id);
        }
      }
    }
    
    // Process Face
    if (faceLandmarker) {
      const faceResults = faceLandmarker.detectForVideo(video, startTimeMs);
      if (faceResults.faceBlendshapes && faceResults.faceBlendshapes.length > 0) {
        const blendshapes = faceResults.faceBlendshapes[0].categories;
        
        // Look for jawOpen to detect breathing/mouth open
        const jawOpen = blendshapes.find(b => b.categoryName === 'jawOpen')?.score || 0;
        
        // Lower threshold to make it trigger faster and more easily
        if (jawOpen > 0.08) {
          // Mouth is open, get mouth center
          const landmarks = faceResults.faceLandmarks[0];
          // 13 and 14 are upper and lower lip inner
          const upperLip = landmarks[13];
          const lowerLip = landmarks[14];
          
          const mouthX = (upperLip.x + lowerLip.x) / 2;
          const mouthY = (upperLip.y + lowerLip.y) / 2;
          
          // Add fog!
          addBreathFog(mouthX, mouthY, jawOpen);
        }
      }
    }
  }
  
  window.requestAnimationFrame(renderLoop);
}

async function main() {
  try {
    await setupCamera();
    video.play();
    
    await initializeMediaPipe();
    
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    
    loadingScreen.classList.add('hidden');
    instructionsScreen.classList.remove('hidden');
    
    // Hide instructions after 5 seconds
    setTimeout(() => {
      instructionsScreen.classList.add('hidden');
    }, 5000);
    
    renderLoop();
  } catch (error) {
    console.error("Error setting up magic mirror:", error);
    loadingScreen.innerHTML = `<p>Error: ${error.message}</p>`;
  }
}

main();
