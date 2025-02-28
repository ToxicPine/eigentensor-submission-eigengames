import { NextRequest, NextResponse } from 'next/server'

export async function POST(request: NextRequest) {
  try {
    // Parse the multipart form data
    const formData = await request.formData()
    const imageFile = formData.get('image') as File
    
    if (!imageFile) {
      return NextResponse.json({ 
        status: 'error', 
        message: 'No Image Provided' 
      }, { status: 400 })
    }
    
    // Convert the file to an ArrayBuffer
    const imageBuffer = await imageFile.arrayBuffer()
    
    // Convert ArrayBuffer to base64 for the ML backend
    // (assuming the ML backend still expects base64)
    const base64Image = Buffer.from(imageBuffer).toString('base64')
    
    // Send image to ML backend
    const response = await fetch('http://localhost:8989/infer', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        image_bytes: base64Image,
        mode: 'regular'
      })
    })

    const result = await response.json()

    if (result.status === 'error' || !response.ok) {
      return NextResponse.json({ status: 'error', message: result.message }, { status: 400 })
    }

    return NextResponse.json({
      status: 'success', 
      prediction: result.data
    })

  } catch (error) {
    console.error('Error calling ML backend:', error)
    return NextResponse.json({ 
      status: 'error',
      message: 'Failed to get prediction'
    }, { status: 500 })
  }
}