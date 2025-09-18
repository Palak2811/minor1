import jwt from "jsonwebtoken";

export const requireAuth = (roles = []) => {
  const rolesArr = Array.isArray(roles) ? roles : [roles];
  return (req, res, next) => {
    try {
      const token = req.headers.authorization?.split(" ")[1] || req.cookies?.token;
      if (!token) return res.status(401).json({ error: "Unauthorized" });
      const payload = jwt.verify(token, process.env.JWT_SECRET || "dev_secret");
      req.user = payload; // { id, role }
      if (rolesArr.length && !rolesArr.includes(payload.role)) {
        return res.status(403).json({ error: "Forbidden" });
      }
      next();
    } catch (e) {
      return res.status(401).json({ error: "Invalid token" });
    }
  }
}
