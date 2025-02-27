'use client'

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'

interface IntroDialogProps {
  onComplete: () => void
}

const introPages = [
  {
    title: "Welcome to the EigenTensor Demo",
    description: "EigenTensor is the easiest way to deploy work onto a remote GPU ever. Oh, and the results are verifiable!"
  },
  {
    title: "What is EigenTensor?", 
    description: "EigenTensor is a universal, memory-safe, cross-platform format and runtime for GPU computations. It makes deploying GPU work easy and flexible for EigenLayer AVSes. It gives developers access to the full extent of each GPU's capabilities without permitting malicious code execution."
  },
  {
    title: "Why Should I Care?",
    description: "Unlike other approaches that saddle you with technical debt, EigenTensor makes GPU deployment both simple and flexible. Its graph-based computations can be efficiently split between devices to reduce resource needs, while still ensuring results are verified through consensus."
  },
  {
    title: "How Does It Work?",
    description: "We reverse-engineered TinyGrad, a popular ML library that builds executable tensor operation graphs for GPUs. We found a way to inject placeholder values into these graphs through its tensor buffer handling. This lets us define computations once and run them anywhere - perfect for distributed GPU work."
  },
  {
    title: "How to Use",
    description: "Draw a number between 0-9. No pressure, but our AI has very high standards and judges your abilities harshly. It's not because the ML model is bad, I promise!"
  },
  {
    title: "Behind the Scenes",
    description: "When you submit your drawing, it gets sent to a server which then forwards it to an EigenTensor-enabled GPU somewhere. Yes, we have fancy consensus verification between nodes, but let's be real - for this demo, we just want to see if it can recognize your number!"
  }
]

export default function IntroDialog({ onComplete }: IntroDialogProps) {
  const [currentPage, setCurrentPage] = useState(0)
  const isLastPage = currentPage === introPages.length - 1
  const isFirstPage = currentPage === 0

  const handleNext = () => {
    if (isLastPage) {
      onComplete()
    } else {
      setCurrentPage(prev => prev + 1)
    }
  }

  const handleBack = () => {
    setCurrentPage(prev => prev - 1)
  }

  return (
    <Dialog open={true} onOpenChange={() => {}}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader className="mt-6 mb-2">
          <DialogTitle className="text-2xl font-bold mb-2">{introPages[currentPage].title}</DialogTitle>
          <DialogDescription className="text-md text-gray-600">
            {introPages[currentPage].description}
          </DialogDescription>
        </DialogHeader>
        <DialogFooter className="sm:justify-between flex gap-4">
          {!isFirstPage && (
            <Button onClick={handleBack} variant="outline" className="w-full grow">
              Back
            </Button>
          )}
          <Button onClick={handleNext} className="w-full grow">
            {isLastPage ? "Let's Go!" : "Next"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}