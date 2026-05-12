import { motion } from "framer-motion"
import { useNavigate } from "react-router-dom"

function LandingPage() {

  const navigate = useNavigate()

  return (
    <div
      style={{
        minHeight:"100vh",
        background:
          "linear-gradient(135deg, #09090f 40%, #3d0b2e 100%)",
        display:"flex",
        justifyContent:"center",
        alignItems:"center",
        padding:"40px"
      }}
    >

      <motion.div
        initial={{ y:-100, opacity:0 }}
        animate={{ y:0, opacity:1 }}
        transition={{ duration:1 }}
        style={{
          textAlign:"center",
          maxWidth:"800px"
        }}
      >

        <h1
          style={{
            fontSize:"4rem",
            marginBottom:"20px",
            fontWeight:"bold"
          }}
        >
          Network Intrusion Detection System
        </h1>

        <p
          style={{
            fontSize:"1.3rem",
            lineHeight:"1.8",
            color:"#d0d0d0",
            marginBottom:"40px"
          }}
        >
          AI Powered Cybersecurity Platform that detects malicious
          network traffic using Machine Learning and Deep Analysis.
        </p>

        <motion.button
          whileHover={{ scale:1.05 }}
          whileTap={{ scale:0.95 }}
          onClick={() => navigate("/dashboard")}
          style={{
            padding:"16px 40px",
            border:"none",
            borderRadius:"12px",
            background:"#ff2f92",
            color:"white",
            fontSize:"1.1rem",
            fontWeight:"bold"
          }}
        >
          Welcome
        </motion.button>

      </motion.div>

    </div>
  )
}

export default LandingPage