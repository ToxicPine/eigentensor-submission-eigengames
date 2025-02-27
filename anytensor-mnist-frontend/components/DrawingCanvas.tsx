'use client'

import { useEffect, useRef, useState } from 'react'

interface DrawingCanvasProps {
  onDrawingComplete: () => void
  setCanvasRef: (ref: HTMLCanvasElement | null) => void
}

export default function DrawingCanvas({ onDrawingComplete, setCanvasRef }: DrawingCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isDrawing, setIsDrawing] = useState(false)
  const [hasDrawnSomething, setHasDrawnSomething] = useState(false)
  
  const canvasSize = 512 // Size in pixels

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    setCanvasRef(canvas)
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Set up canvas
    ctx.lineWidth = 24
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    ctx.strokeStyle = 'black'
    
    // Set up event listeners
    const handleMouseDown = (e: MouseEvent) => {
      setIsDrawing(true)
      setHasDrawnSomething(true)
      
      const rect = canvas.getBoundingClientRect()
      const x = e.clientX - rect.left
      const y = e.clientY - rect.top
      
      ctx.beginPath()
      ctx.moveTo(x, y)
    }
    
    const handleMouseMove = (e: MouseEvent) => {
      if (!isDrawing) return
      
      const rect = canvas.getBoundingClientRect()
      const x = e.clientX - rect.left
      const y = e.clientY - rect.top
      
      ctx.lineTo(x, y)
      ctx.stroke()
    }
    
    const handleMouseUp = () => {
      if (isDrawing) {
        setIsDrawing(false)
        
        // Pixelate the drawing when finger is lifted
        pixelateCanvas(canvas, 16)
        
        // Notify parent that drawing is complete
        onDrawingComplete()
      }
    }
    
    // Add event listeners
    canvas.addEventListener('mousedown', handleMouseDown)
    canvas.addEventListener('mousemove', handleMouseMove)
    canvas.addEventListener('mouseup', handleMouseUp)
    canvas.addEventListener('mouseleave', handleMouseUp)
    
    // Add touch support
    canvas.addEventListener('touchstart', (e) => {
      e.preventDefault()
      const touch = e.touches[0]
      const mouseEvent = new MouseEvent('mousedown', {
        clientX: touch.clientX,
        clientY: touch.clientY
      })
      canvas.dispatchEvent(mouseEvent)
    })
    
    canvas.addEventListener('touchmove', (e) => {
      e.preventDefault()
      const touch = e.touches[0]
      const mouseEvent = new MouseEvent('mousemove', {
        clientX: touch.clientX,
        clientY: touch.clientY
      })
      canvas.dispatchEvent(mouseEvent)
    })
    
    canvas.addEventListener('touchend', (e) => {
      e.preventDefault()
      const mouseEvent = new MouseEvent('mouseup')
      canvas.dispatchEvent(mouseEvent)
    })
    
    // Cleanup
    return () => {
      canvas.removeEventListener('mousedown', handleMouseDown)
      canvas.removeEventListener('mousemove', handleMouseMove)
      canvas.removeEventListener('mouseup', handleMouseUp)
      canvas.removeEventListener('mouseleave', handleMouseUp)
      
      canvas.removeEventListener('touchstart', (e) => e.preventDefault())
      canvas.removeEventListener('touchmove', (e) => e.preventDefault())
      canvas.removeEventListener('touchend', (e) => e.preventDefault())
    }
  }, [isDrawing, onDrawingComplete, setCanvasRef])

  // Function to pixelate the canvas in a 12x12 grid with 4-bit color
  const pixelateCanvas = (canvas: HTMLCanvasElement, gridSize: number) => {
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    const w = canvas.width
    const h = canvas.height
    
    // Create an empty canvas for pixelated result
    const tempCanvas = document.createElement('canvas')
    tempCanvas.width = w
    tempCanvas.height = h
    const tempCtx = tempCanvas.getContext('2d')
    if (!tempCtx) return
    
    // Draw original image to temp canvas
    tempCtx.drawImage(canvas, 0, 0)
    
    // Clear original canvas
    ctx.clearRect(0, 0, w, h)
    
    // Calculate grid cell size
    const cellW = w / gridSize
    const cellH = h / gridSize
    
    // Process each grid cell
    for (let y = 0; y < gridSize; y++) {
      for (let x = 0; x < gridSize; x++) {
        // Calculate the cell position
        const cellX = Math.floor(x * cellW)
        const cellY = Math.floor(y * cellH)
        
        // Get pixel data for the entire cell area
        const cellData = tempCtx.getImageData(cellX, cellY, Math.ceil(cellW), Math.ceil(cellH))
        const data = cellData.data
        
        // Calculate average color values for the cell
        let totalR = 0, totalG = 0, totalB = 0, totalA = 0
        let drawnPixels = 0
        
        for (let i = 0; i < data.length; i += 4) {
          // Count pixels with any opacity as drawn pixels
          if (data[i+3] > 0) {
            drawnPixels++
            totalR += data[i]
            totalG += data[i+1]
            totalB += data[i+2]
            totalA += data[i+3]
          }
        }
        
        // Only fill the cell if it contains any drawing (threshold can be adjusted)
        if (drawnPixels > 0) {
          // Calculate average colors of drawn pixels
          const avgR = Math.round(totalR / drawnPixels)
          const avgG = Math.round(totalG / drawnPixels)
          const avgB = Math.round(totalB / drawnPixels)
          const avgA = Math.round(totalA / drawnPixels)
          
          // Convert to 4-bit color (16 levels)
          const r = Math.round(avgR / 16) * 16
          const g = Math.round(avgG / 16) * 16
          const b = Math.round(avgB / 16) * 16
          const a = Math.round(avgA / 16) * 16
          
          // Fill the cell with the pixelated color
          ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${a / 255})`
          ctx.fillRect(cellX, cellY, cellW, cellH)
        }
      }
    }
  }

  return (
    <div className="relative border-2 rounded-md shadow-sm">
      <canvas
        ref={canvasRef}
        width={canvasSize}
        height={canvasSize}
        className="bg-white touch-none rounded-md"
      />
      {!hasDrawnSomething && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <p className="text-gray-300 text-lg">use your mouse to draw</p>
        </div>
      )}
    </div>
  )
} 