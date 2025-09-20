import { Router } from "express";
import PatientProfile from "../models/PatientProfile.js";
import QuestionnaireResponse from "../models/QuestionnaireResponse.js";
import { requireAuth } from "../middleware/auth.js";

const router = Router();

// Assign patient to doctor
router.post("/assign", requireAuth("doctor"), async (req, res) => {
  const { patientUserId } = req.body;
  const updated = await PatientProfile.findOneAndUpdate(
    { user: patientUserId },
    { $set: { assignedDoctor: req.user.id } },
    { new: true }
  );
  res.json({ assigned: updated });
});

// My patients
router.get("/patients", requireAuth("doctor"), async (req, res) => {
  const patients = await PatientProfile.find({ assignedDoctor: req.user.id }).populate("user", "name email");
  res.json({ patients });
});

// View latest questionnaire per patient
router.get("/patients/latest", requireAuth("doctor"), async (req, res) => {
  const patients = await PatientProfile.find({ assignedDoctor: req.user.id }).select("user");
  const ids = patients.map(p => p.user);
  const latest = await QuestionnaireResponse.aggregate([
    { $match: { patient: { $in: ids } } },
    { $sort: { createdAt: -1 } },
    { $group: {
      _id: "$patient",
      doc: { $first: "$$ROOT" }
    }}
  ]);
  res.json({ latest });
});

export default router;
