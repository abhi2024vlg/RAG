"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { Send } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Badge } from "@/components/ui/badge"

type Message = {
  id: string
  content: string
  isUser: boolean
  source?: string
}

export default function ChatApp() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const scrollAreaRef = useRef<HTMLDivElement>(null)

  // Scroll to bottom when messages change
  useEffect(() => {
    if (scrollAreaRef.current) {
      const scrollContainer = scrollAreaRef.current
      scrollContainer.scrollTop = scrollContainer.scrollHeight
    }
  }, [messages])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!input.trim()) return

    // Add user message to chat
    const userMessage: Message = {
      id: Date.now().toString(),
      content: input,
      isUser: true,
    }

    setMessages((prev) => [...prev, userMessage])
    setInput("")
    setIsLoading(true)

    try {
      // Send request to backend
      const response = await fetch("/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: input }),
      })

      if (!response.ok) {
        throw new Error("Failed to get response")
      }

      const data = await response.json()

      // Add bot response to chat
      const botMessage: Message = {
        id: Date.now().toString(),
        content: data.response,
        isUser: false,
        source: data.source,
      }

      setMessages((prev) => [...prev, botMessage])
    } catch (error) {
      console.error("Error:", error)

      // Add error message
      const errorMessage: Message = {
        id: Date.now().toString(),
        content: "Sorry, I couldn't process your request. Please try again.",
        isUser: false,
      }

      setMessages((prev) => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="flex items-center justify-center min-h-screen p-4 bg-muted/30">
      <Card className="w-full max-w-2xl h-[80vh] flex flex-col">
        <CardHeader className="px-6 py-4 border-b">
          <CardTitle>AI Assistant</CardTitle>
        </CardHeader>

        <ScrollArea ref={scrollAreaRef} className="flex-1 p-6">
          <div className="space-y-4">
            {messages.length === 0 ? (
              <div className="text-center text-muted-foreground py-8">Send a message to start the conversation</div>
            ) : (
              messages.map((message) => (
                <div key={message.id} className={`flex ${message.isUser ? "justify-end" : "justify-start"}`}>
                  <div className={`flex gap-3 max-w-[80%] ${message.isUser ? "flex-row-reverse" : "flex-row"}`}>
                    <Avatar className={message.isUser ? "bg-primary" : "bg-secondary"}>
                      <AvatarFallback>{message.isUser ? "U" : "AI"}</AvatarFallback>
                    </Avatar>

                    <div className="flex flex-col gap-1">
                      <div
                        className={`rounded-lg px-4 py-2 ${
                          message.isUser
                            ? "bg-primary text-primary-foreground"
                            : "bg-secondary text-secondary-foreground"
                        }`}
                      >
                        {message.content}
                      </div>

                      {message.source && (
                        <a
                          href={message.source}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-xs text-blue-500 hover:underline ml-2"
                        >
                          <Badge variant="outline" className="gap-1 text-xs">
                            Source
                          </Badge>
                        </a>
                      )}
                    </div>
                  </div>
                </div>
              ))
            )}

            {isLoading && (
              <div className="flex justify-start">
                <div className="flex gap-3 max-w-[80%]">
                  <Avatar className="bg-secondary">
                    <AvatarFallback>AI</AvatarFallback>
                  </Avatar>
                  <div className="flex items-center rounded-lg px-4 py-2 bg-secondary text-secondary-foreground">
                    <div className="flex space-x-1">
                      <div className="h-2 w-2 bg-current rounded-full animate-bounce [animation-delay:-0.3s]"></div>
                      <div className="h-2 w-2 bg-current rounded-full animate-bounce [animation-delay:-0.15s]"></div>
                      <div className="h-2 w-2 bg-current rounded-full animate-bounce"></div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </ScrollArea>

        <CardFooter className="p-4 border-t">
          <form onSubmit={handleSubmit} className="flex w-full gap-2">
            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type your message..."
              disabled={isLoading}
              className="flex-1"
            />
            <Button type="submit" size="icon" disabled={isLoading}>
              <Send className="h-4 w-4" />
            </Button>
          </form>
        </CardFooter>
      </Card>
    </div>
  )
}

