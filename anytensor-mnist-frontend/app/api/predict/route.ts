import { NextRequest, NextResponse } from 'next/server'

export async function POST(request: NextRequest) {
  try {
    const { image } = await request.json()
    
    // Here you would normally send the image to your ML backend
    // For this example, we'll simulate a response with a random number
    
    // Simulate API call delay
    await new Promise(resolve => setTimeout(resolve, 1500))
    
    // 10% chance of error for demonstration
    if (Math.random() < 0.1) {
      return NextResponse.json({ status: 'error' })
    }
    
    // Random prediction between 0-9
    const prediction = Math.floor(Math.random() * 10)
    
    return NextResponse.json({ 
      status: 'success',
      prediction
    })
  } catch (error) {
    return NextResponse.json({ status: 'error' }, { status: 500 })
  }
} 