const express = require('express');
const multer = require('multer');
const path = require('path');
const { exec } = require('child_process');

const app = express();
const PORT = 3000;

// Configure multer for file upload
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, 'uploads/');
    },
    filename: (req, file, cb) => {
        cb(null, Date.now() + path.extname(file.originalname));
    }
});

const upload = multer({ storage: storage });7

// Serve static files from the 'public' directory
app.use(express.static('public'));

// Process uploaded images
app.post('/upload', upload.single('image'), (req, res) => {
    const windowSize = req.body['window-size'];
    const imagePath = req.file.path;
    const processedImagePath = path.join(__dirname, 'public', 'processed_image.jpg');

    // Call the CUDA executable
    //only for backend testing
    // exec(`./median_filter ${imagePath} ${processedImagePath} ${windowSize}`, (error, stdout, stderr) => {
    //     if (error) {
    //         console.error(`exec error: ${error}`);
    //         return res.status(500).send('Error processing image.');
    //     }

    //     res.sendFile(processedImagePath, (err) => {
    //         if (err) {
    //             res.status(500).send(err);
    //         }
    //     });
    // });
    exec(`median_filter.exe ${imagePath} ${processedImagePath} ${windowSize}`, (error, stdout, stderr) => {
        if (error) {
            console.error(`exec error: ${error}`);
            return res.status(500).send('Error processing image.');
        }
        
        console.log(`stdout: ${stdout}`); // Optionally log the output
        res.sendFile(processedImagePath, (err) => {
            if (err) {
                res.status(500).send(err);
            }
        });
    });
});

// Start the server
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
