import React from "react"

function Navbar() {

  return (
    <div className="topbar">
      <div className="brand-title"></div>
      <div className="topbar-actions">
        <a href="/format.csv" download className="btn-primary" aria-label="Download format CSV">
          Format
        </a>
      </div>
    </div>
  )
}

export default Navbar