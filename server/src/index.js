import express from "express";
import mongoose from "mongoose";
import cors from "cors";
import dotenv from "dotenv";
import cookieParser from "cookie-parser";

import authRoutes from "./routes/auth.js";
import patientRoutes from "./routes/patient.js";
import doctorRoutes from "./routes/doctor.js";

dotenv.config();
const app = express();

app.use(cors({
  origin: [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5500",
    "http://localhost:5500"
  ],
  credentials: true
}));

app.use(express.json());
app.use(cookieParser());

// Healthcheck
app.get("/api/health", (_req, res) => res.json({ ok: true }));

app.use("/api/auth", authRoutes);
app.use("/api/patient", patientRoutes);
app.use("/api/doctor", doctorRoutes);

const MONGO_URI = process.env.MONGO_URI || "mongodb://127.0.0.1:27017/pregnancy_app";
const PORT = process.env.PORT || 5000;

mongoose.connect(MONGO_URI).then(() => {
  app.listen(PORT, () => console.log(`Server running on ${PORT}`));
}).catch(err => {
  console.error("Mongo connect error", err);
  process.exit(1);
});
