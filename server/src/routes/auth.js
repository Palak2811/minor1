import { Router } from "express";
import bcrypt from "bcrypt";
import jwt from "jsonwebtoken";
import User from "../models/User.js";
import PatientProfile from "../models/PatientProfile.js";
import DoctorProfile from "../models/DoctorProfile.js";

const router = Router();
const signToken = (user) =>
  jwt.sign({ id: user._id, role: user.role }, process.env.JWT_SECRET || "dev_secret", { expiresIn: "7d" });

router.post("/register", async (req, res) => {
  try {
    const { name, email, password, role } = req.body;
    if (!["patient","doctor"].includes(role)) return res.status(400).json({ error: "Invalid role" });
    const exists = await User.findOne({ email });
    if (exists) return res.status(409).json({ error: "Email already registered" });
    const passwordHash = await bcrypt.hash(password, 10);
    const user = await User.create({ name, email, passwordHash, role });
    if (role === "patient") await PatientProfile.create({ user: user._id });
    if (role === "doctor") await DoctorProfile.create({ user: user._id });
    const token = signToken(user);
    res.json({ token, user: { id: user._id, name, email, role } });
  } catch (e) {
    res.status(500).json({ error: "Registration failed" });
  }
});

router.post("/login", async (req, res) => {
  try {
    const { email, password, role } = req.body;
    const user = await User.findOne({ email, role });
    if (!user) return res.status(401).json({ error: "Invalid credentials" });
    const ok = await bcrypt.compare(password, user.passwordHash);
    if (!ok) return res.status(401).json({ error: "Invalid credentials" });
    const token = signToken(user);
    res.json({ token, user: { id: user._id, name: user.name, email, role: user.role } });
  } catch (e) {
    res.status(500).json({ error: "Login failed" });
  }
});

export default router;
