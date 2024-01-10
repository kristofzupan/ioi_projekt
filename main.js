import  {PoseLandmarker, FilesetResolver, DrawingUtils} from "/node_modules/@mediapipe/tasks-vision/vision_bundle.mjs"
//import * as Tone from '/node_modules/tone/build/Tone.js'

let synths = []
const tones = ["C", "D", "E", "F", "G", "A", "B"]
for (let i = 0; i < 7; i++) {
    synths.push(new Tone.Synth().toDestination())
}
let poseLandmarker = undefined
let runningMode = "LIVE_STREAM"
let enableWebcamButton
let webcamRunning = false
let calibration = false
let isCalibrating = false
let isCalibratingFirstText = false
let isCalibratingSecondText = false
let isStartTriggered = false
let countDownNumber = -1
let enableCalibration = document.getElementById("calibrateButton");
enableCalibration.addEventListener("click", changeCalibration);
const videoWidth = "1280px"
const videoHeight = "720px"

// Before we can use PoseLandmarker class we must wait for it to finish loading
const createPoseLandmarker = async () => {
    const vision = await FilesetResolver.forVisionTasks(
        "/node_modules/@mediapipe/tasks-vision/wasm"
    );
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `/pose_landmarker_heavy.task`,
            delegate: "GPU"
        },
        runningMode: runningMode,
        numPoses: 8,
        minPoseDetectionConfidence: 0.7,
        minPosePresenceConfidence: 0.85,
        minTrackingConfidence: 0.75
    });
};
createPoseLandmarker();


const video = document.getElementById("webcam")
const canvasElement = document.getElementById("output_canvas")
const canvasCtx = canvasElement.getContext("2d");
const drawingUtils = new DrawingUtils(canvasCtx);

// Check if webcam access is supported.
const hasGetUserMedia = () => !!navigator.mediaDevices?.getUserMedia;

// If webcam supported, add event listener to button for when user wants to activate it.
if (hasGetUserMedia()) {
    enableWebcamButton = document.getElementById("webcamButton");
    enableWebcamButton.addEventListener("click", enableCam);
} else {
    console.warn("getUserMedia() is not supported by your browser");
}

// Enable the live webcam view and start detection.
function enableCam(event) {
    Tone.start()

    if (!poseLandmarker) {
        console.log("Wait! poseLandmaker not loaded yet.");
        return;
    }

    if (webcamRunning === true) {
        webcamRunning = false;
        enableWebcamButton.innerText = "VKLOPI SLEDENJE";
    } else {
        webcamRunning = true;
        enableWebcamButton.innerText = "USTAVI SLEDENJE";
    }

    // Activate the webcam stream.
    navigator.mediaDevices.getUserMedia({video: {width: 1280, height: 720}}).then((stream) => {
        video.srcObject = stream;
        video.addEventListener("loadeddata", predictWebcam)
    });
}

function changeCalibration(event) {
    if (calibration) {
        enableCalibration.removeAttribute("disabled")
        calibration = false
        enableCalibration.innerText = "KALIBRACIJA"
    } else {
        isCalibrating = true
        calibration = true
        enableCalibration.setAttribute("disabled", "disabled")
        setTimeout(() => {
            isCalibratingFirstText = true
            setTimeout(() => {
                isCalibratingFirstText = false
                setTimeout(() => {
                    isCalibratingSecondText = true
                    setTimeout(() => {
                        isCalibratingSecondText = false

                        setTimeout(() => {
                            isCalibrating = false
                        }, "1000");
                    }, "4000");
                }, "1000");
            }, "5000");
        }, "2000");
    }
}

function countDown() {
    countDownNumber = 3
    setTimeout(() => {
        countDownNumber = 2
        setTimeout(() => {
            countDownNumber = 1
            setTimeout(() => {
                countDownNumber = 0
                setTimeout(() => {
                    countDownNumber = -1
                    isStartTriggered = false
                    changeCalibration()
                }, "1000");
            }, "1000");
        }, "1000");
    }, "1000");
}

function calculate_hand_height(pose) {
    const shoulders = [pose[11], pose[12]];
    const hands =[pose[15], pose[16]]

    const shoulder_height = (shoulders[0].y + shoulders[1].y) / 2
    const hand_height = (hands[0].y + hands[1].y) / 2

    const height_diff = shoulder_height - hand_height
    return height_diff
}

function color_detection(pose) {
    const shoulders = [pose[11], pose[12]];
    const hips = [pose[23], pose[24]];

    // Calculate the center of the quadrilateral
    const centerX = (shoulders[0].x + shoulders[1].x + hips[0].x + hips[1].x) / 4;
    const centerY = (shoulders[0].y + shoulders[1].y + hips[0].y + hips[1].y) / 4;

    let width = Math.min(Math.abs(hips[0].x - hips[1].x), Math.abs(shoulders[0].x - shoulders[1].x)) * 1280
    let height = Math.min(Math.abs(hips[0].y - shoulders[0].y), Math.abs(hips[1].y - shoulders[1].y)) * 720

    if (width < 2)
        width = 2;
    if (height < 2)
        height = 2;
    // Calculate the bounding box around the upper body
    const upperBodyBoundingBox = {
        x: (centerX*1280) - (width/4),
        y: (centerY*720) - (height/4),
        width: width/2,
        height: height/2,
    };

    // Capture pixel data within the defined region of interest
    const imageData = canvasCtx.getImageData(upperBodyBoundingBox.x, upperBodyBoundingBox.y, upperBodyBoundingBox.width, upperBodyBoundingBox.height);
    const pixelData = imageData.data;

    // Analyze colors (similar to the previous example)
    let totalRed = 0;
    let totalGreen = 0;
    let totalBlue = 0;
    let totalPixels = 0;

    for (let i = 0; i < pixelData.length; i += 4) {
        totalRed += pixelData[i];
        totalGreen += pixelData[i + 1];
        totalBlue += pixelData[i + 2];
        totalPixels++;
    }

    //console.log(totalPixels)
    const averageRed = totalRed / totalPixels;
    const averageGreen = totalGreen / totalPixels;
    const averageBlue = totalBlue / totalPixels;

    const averageColor = `rgb(${averageRed}, ${averageGreen}, ${averageBlue})`;

    // Draw the bounding box (optional, for visualization)
    canvasCtx.fillStyle = `rgba(${averageRed}, ${averageGreen}, ${averageBlue}, 1)`;
    canvasCtx.fillRect(upperBodyBoundingBox.x, upperBodyBoundingBox.y, upperBodyBoundingBox.width, upperBodyBoundingBox.height);

    // Output or use the result
    return [averageRed, averageGreen, averageBlue]
}

function RGBtoHSV(r, g, b) {
    if (arguments.length === 1) {
        g = r.g, b = r.b, r = r.r;
    }
    let max = Math.max(r, g, b), min = Math.min(r, g, b),
        d = max - min,
        h,
        s = (max === 0 ? 0 : d / max),
        v = max / 255;

    switch (max) {
        case min: h = 0; break;
        case r: h = (g - b) + d * (g < b ? 6: 0); h /= 6 * d; break;
        case g: h = (b - r) + d * 2; h /= 6 * d; break;
        case b: h = (r - g) + d * 4; h /= 6 * d; break;
    }
    return h * 355
}

function rgbToNote(r, g, b) {
   let h = RGBtoHSV(r, g, b)

    const colorRanges = {
        'C': { lower: 0, upper:  15},
        'D': { lower: 15, upper:  40},
        'E': { lower: 40, upper:  70},
        'F': { lower: 70, upper:  150},
        'G': { lower: 150, upper:  190},
        'A': { lower: 190, upper:  260},
        'B': { lower: 260, upper:  330},
        'C2': {lower: 330, upper: 355}
    };

    // Check if the input RGB values fall within the defined color ranges
    for (const [color, { lower, upper }] of Object.entries(colorRanges)) {
        if (
            h >= lower && h <= upper
        ) {
            if (color === 'C2')
                return 'C'
            return color;
        }
    }

    // If no color is detected, return a default color or null
    return null;
}

let octave = 4
let lastVideoTime = -1
async function predictWebcam() {
    let temp = document.getElementById("octave").value
    if (temp > 0 && temp < 6)
        octave = temp
    canvasElement.style.height = videoHeight;
    video.style.height = videoHeight;
    canvasElement.style.width = videoWidth;
    video.style.width = videoWidth;

    // Now let's start detecting the stream.
    let startTimeMs = performance.now();
    if (lastVideoTime !== video.currentTime) {
        poseLandmarker.detectForVideo(video, startTimeMs, (result) => {
            canvasCtx.drawImage(video, 0, 0,)
            //console.log(result)
            let areAllHandsUp = true
            for (let landmark of result.landmarks) {
                drawingUtils.drawLandmarks(landmark, {
                    radius: (data) => DrawingUtils.lerp(data.from.z, -0.15, 0.1, 5, 1)
                });
                drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS);

                let rgb = color_detection(landmark)
                let tone = rgbToNote(rgb[0], rgb[1], rgb[2])
                canvasCtx.font = "30px Arial"
                canvasCtx.fillStyle = "white"
                canvasCtx.fillText(tone,landmark[24].x*1280 + 20, landmark[24].y*720 - 20)
                if (calibration && lastVideoTime !== -1) {
                    let handHeight = calculate_hand_height(landmark)
                    if (handHeight < 0) {
                        areAllHandsUp = false
                    }
                } else {
                    let volume = calculate_hand_height(landmark)
                    const now = Tone.now()
                    const synth_index = tones.indexOf(tone);
                    if (volume > 0 && synths[synth_index].envelope.value < 0.035 && synths[synth_index].envelope.value !== 1e-7) { // && synths[synth_index].envelope.value !== 1e-7
                        console.log(tone, octave)
                        synths[synth_index].triggerAttackRelease(tone + octave, '8n', now);
                    }
                }
            }
            if (lastVideoTime === -1) {
                changeCalibration(null)
            }
            if (areAllHandsUp && !isCalibrating && !isStartTriggered && countDownNumber === -1 && calibration) {
                isStartTriggered = true
                console.log("TRIGGER")
                countDown()
            }

            if (isStartTriggered) {
                canvasCtx.fillStyle = "rgba(0, 0, 0, 0.6)";
                canvasCtx.fillRect(0, 0, 1280, 720);

                canvasCtx.font = "48px Arial"
                canvasCtx.fillStyle = "white"
                if (countDownNumber === 0) {
                    canvasCtx.fillText("START",0.43*1280, 0.5*720)
                } else {
                    canvasCtx.fillText(countDownNumber,0.48*1280, 0.5*720)
                }
            }


            if (isCalibratingFirstText) {
                canvasCtx.fillStyle = "rgba(0, 0, 0, 0.6)";
                canvasCtx.fillRect(0, 0, 1280, 720);

                canvasCtx.font = "48px Arial"
                canvasCtx.fillStyle = "white"
                canvasCtx.fillText("PRIPRAVITE SE! ",0.35*1280, 0.5*720 - 60)
                canvasCtx.fillText("VSI KI ŽELITE IGRATI BODITE POZORNI DA",0.1*1280, 0.5*720)
                canvasCtx.fillText("VAS SISTEM VIDI IN ZAZNA VAŠO POZO!",0.125*1280, 0.5*720 + 60)
            }
            if (isCalibratingSecondText) {
                canvasCtx.fillStyle = "rgba(0, 0, 0, 0.6)";
                canvasCtx.fillRect(0, 0, 1280, 720);

                canvasCtx.font = "48px Arial"
                canvasCtx.fillStyle = "white"
                canvasCtx.fillText("KO STE PRIPRAVLJENI VSI DVIGNITE ROKE!",0.1*1280, 0.5*720)
            }
            lastVideoTime = video.currentTime;
            canvasCtx.restore();
        });
    }

    // Call this function again to keep predicting when the browser is ready.
    if (webcamRunning === true) {
        window.requestAnimationFrame(predictWebcam);
    }
}