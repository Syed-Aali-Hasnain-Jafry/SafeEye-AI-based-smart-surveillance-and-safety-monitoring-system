
import { useEffect, useState } from 'react'
import './App.css'

function App() {
  const [activePage, setActivePage] = useState('Dashboard')
  const [videoFile, setVideoFile] = useState(null)
  const [videoUrl, setVideoUrl] = useState(null)

useEffect(() => {
  if (!videoFile) {
    setVideoUrl(null)
    return
  }

  const url = URL.createObjectURL(videoFile)
  setVideoUrl(url)

  return () => {
    URL.revokeObjectURL(url)
  }
}, [videoFile])

function handleVideoChange(event) {
  const file = event.target.files?.[0]

  if (!file) {
    return
  }

  setVideoFile(file)
}
  const menuItems = [
    'Dashboard',
    'Cameras',
    'Person Tracking',
    'Person Search',
    'Incidents',
  ]

  return (
    <div className="dashboard-layout">

      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <h2>SafEye</h2>
          <p>Smart Surveillance System</p>
        </div>

        <nav className="sidebar-navigation">
          {menuItems.map((item) => (
            <button
              key={item}
              className={
                activePage === item
                  ? 'nav-item active'
                  : 'nav-item'
              }
              onClick={() => setActivePage(item)}
            >
              {item}
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">
          <p>SafEye FYP</p>
          <span>Bahria University Islamabad</span>
        </div>
      </aside>

      {/* Main content */}
      <main className="main-content">

        {/* Header */}
        <header className="top-header">
          <div>
            <h1>{activePage}</h1>
            <p>Intelligent Person Tracking and Incident Detection</p>
          </div>

          <div className="system-status">
            <span className="status-indicator"></span>
            Frontend Demo
          </div>
        </header>

        {/* Dashboard statistics */}
        <section className="stats-grid">

          <div className="stat-card">
            <h3>Connected Cameras</h3>
            <p>0</p>
            <span>No camera source connected</span>
          </div>

          <div className="stat-card">
            <h3>Active Person Tracks</h3>
            <p>0</p>
            <span>Tracking engine not connected</span>
          </div>

          <div className="stat-card">
            <h3>Incidents</h3>
            <p>0</p>
            <span>Incident module pending integration</span>
          </div>

        </section>

        {/* Camera monitoring */}
        <section className="camera-section">
          <div className="section-heading">
            <h2>Camera Monitoring</h2>
            <span>Demo Mode</span>
          </div>

      
<div className="camera-preview">

  {videoUrl ? (
    <div className="video-container">

      <video
        key={videoUrl}
        src={videoUrl}
        controls
        className="camera-video"
      >
        Your browser does not support video playback.
      </video>

      <div className="video-information">
        <span>{videoFile.name}</span>

        <button
          type="button"
          onClick={() => setVideoFile(null)}
        >
          Remove Video
        </button>
      </div>

    </div>
  ) : (
    <div className="camera-placeholder">

      <h3>No Video Source Selected</h3>

      <p>
        Select a video from your computer
        to preview it in SafEye.
      </p>

      <label
        htmlFor="video-upload"
        className="upload-button"
      >
        Select Video
      </label>

      <input
        id="video-upload"
        type="file"
        accept="video/mp4,video/webm,video/ogg"
        onChange={handleVideoChange}
        className="video-input"
      />

    </div>
  )}

</div>

        </section>

        {/* Person tracking placeholder */}
        <section className="tracking-section">
          <h2>Person Tracking</h2>

          <div className="tracking-placeholder">
            <p>
              Detection boxes, tracking IDs, and person
              movement information will appear here
              after YOLO + ByteTrack integration.
            </p>
          </div>
        </section>

      </main>
    </div>
  )
}

export default App
