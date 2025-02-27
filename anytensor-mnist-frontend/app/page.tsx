'use client'

import { useEffect, useState } from 'react'
import { getCookie, setCookie } from 'cookies-next'
import IntroDialog from '@/components/IntroDialog'
import DrawingCanvas from '@/components/DrawingCanvas'
import { Button } from '@/components/ui/button'
import ResultView from '@/components/ResultView'
import { Loader2 } from 'lucide-react'

export default function Home() {
  const [showIntro, setShowIntro] = useState(false)
  const [hasDrawn, setHasDrawn] = useState(false)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [result, setResult] = useState<{status: string; prediction?: number} | null>(null)
  const [canvasRef, setCanvasRef] = useState<HTMLCanvasElement | null>(null)
  
  useEffect(() => {
    // Check if the intro has been shown before
    const hasSeenIntro = getCookie('hasSeenIntro')
    if (!hasSeenIntro) {
      setShowIntro(true)
    }
  }, [])

  const handleIntroComplete = () => {
    setCookie('hasSeenIntro', 'true', { maxAge: 60 * 60 * 24 * 365 })
    setShowIntro(false)
  }

  const handleDrawingComplete = () => {
    setHasDrawn(true)
  }

  const handleClear = () => {
    if (canvasRef) {
      const ctx = canvasRef.getContext('2d')
      if (ctx) {
        ctx.clearRect(0, 0, canvasRef.width, canvasRef.height)
        setHasDrawn(false)
      }
    }
  }

  const handleSubmit = async () => {
    if (!canvasRef) return

    setIsSubmitting(true)
    
    try {
      // Convert canvas to base64 image
      const imageData = canvasRef.toDataURL('image/png')
      
      // Send to API
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: imageData }),
      })
      
      const data = await response.json()
      setResult(data)
    } catch (error) {
      console.error('Error Submitting Drawing:', error)
      setResult({ status: 'error' })
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleTryAgain = () => {
    setResult(null)
    handleClear()
  }

  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-4 bg-white">
      {showIntro && <IntroDialog onComplete={handleIntroComplete} />}
      
      {!showIntro && !result && (
        <div className="flex flex-col items-center gap-4">
          <DrawingCanvas 
            onDrawingComplete={handleDrawingComplete} 
            setCanvasRef={setCanvasRef}
          />
          
          {hasDrawn && !isSubmitting && (
            <div className="flex flex-col gap-4 mt-4 w-full max-w-full">
              <Button variant="outline" onClick={handleClear} className="w-full grow">
                Clear
              </Button>
              <Button onClick={handleSubmit} className="w-full grow">
                Submit
              </Button>
            </div>
          )}
          
          {isSubmitting && (
            <div className="flex flex-col items-center gap-2 mt-4">
              <Loader2 className="h-8 w-8 animate-spin" />
              <p className="text-sm text-gray-500">Processing your drawing...</p>
            </div>
          )}
        </div>
      )}
      
      {result && (
        <ResultView 
          result={result} 
          onTryAgain={handleTryAgain}
        />
      )}
    </main>
  )
}
