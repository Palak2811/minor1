import mongoose from "mongoose";

const questionnaireResponseSchema = new mongoose.Schema({
  patient: { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true },
  answers: {
    nausea: { type: String, enum: ["none","mild","moderate","severe"] },
    bleeding: { type: String, enum: ["no","spotting","heavy"] },
    fetalKicksLast2Hrs: { type: Number, min: 0 },
    systolicBP: { type: Number, min: 50, max: 240 },
    diastolicBP: { type: Number, min: 30, max: 150 },
    fastingGlucose: { type: Number, min: 40, max: 400 },
    swellingFeet: { type: String, enum: ["no","yes"] },
    headache: { type: String, enum: ["no","mild","severe"] },
  },
  provisionalRiskScore: { type: Number, min: 0, max: 1 }, // placeholder before ML
  provisionalRiskLabel: { type: String, enum: ["Low","Moderate","High"] },
}, { timestamps: true });

export default mongoose.model("QuestionnaireResponse", questionnaireResponseSchema);
