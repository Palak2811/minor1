import { Router } from "express";
import PatientProfile from "../models/PatientProfile.js";
import QuestionnaireResponse from "../models/QuestionnaireResponse.js";
import { requireAuth } from "../middleware/auth.js";

const router = Router();

// Simple provisional scoring before ML
const scoreProvisional = (a) => {
  let s = 0;
  if (a.bleeding === "heavy") s += 0.7;
  else if (a.bleeding === "spotting") s += 0.3;
  if ((a.systolicBP >= 140) || (a.diastolicBP >= 90)) s += 0.3;
  if (a.fastingGlucose >= 126) s += 0.2;
  if (a.fetalKicksLast2Hrs < 10) s += 0.3;
  if (a.headache === "severe") s += 0.2;
  if (a.swellingFeet === "yes") s += 0.1;
  s = Math.min(1, s);
  const label = s < 0.3 ? "Low" : s < 0.6 ? "Moderate" : "High";
  return { s, label };
};

// Get own profile
router.get("/me", requireAuth("patient"), async (req, res) => {
  const profile = await PatientProfile.findOne({ user: req.user.id }).populate("assignedDoctor", "name email");
  res.json({ profile });
});

// Update profile
router.put("/me", requireAuth("patient"), async (req, res) => {
  const profile = await PatientProfile.findOneAndUpdate(
    { user: req.user.id },
    { $set: req.body },
    { new: true }
  );
  res.json({ profile });
});

// Submit questionnaire
router.post("/questionnaire", requireAuth("patient"), async (req, res) => {
  const answers = req.body?.answers || {};
  const { s, label } = scoreProvisional(answers);
  const doc = await QuestionnaireResponse.create({
    patient: req.user.id,
    answers,
    provisionalRiskScore: s,
    provisionalRiskLabel: label
  });
  res.json({ result: { riskScore: s, riskLabel: label }, id: doc._id });
});

// List my submissions
router.get("/questionnaire", requireAuth("patient"), async (req, res) => {
  const list = await QuestionnaireResponse.find({ patient: req.user.id }).sort({ createdAt: -1 });
  res.json({ submissions: list });
});

export default router;
