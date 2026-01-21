import { useEffect, useState, useRef } from 'react';
import { Send, User, Bot, Loader2, Plus } from 'lucide-react';
import { arquetiposApi, interaccionApi, type Arquetipo } from '../lib/api';

interface Message {
  id: string;
  rol: 'usuario' | 'sintetico';
  contenido: string;
  fricciones?: string[];
  emociones?: Record<string, unknown>;
}

export function Interaccion() {
  const [arquetipos, setArquetipos] = useState<Arquetipo[]>([]);
  const [selectedArquetipo, setSelectedArquetipo] = useState<string>('');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    loadArquetipos();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  async function loadArquetipos() {
    try {
      const { arquetipos } = await arquetiposApi.list();
      setArquetipos(arquetipos);
    } catch (error) {
      console.error('Error loading arquetipos:', error);
    }
  }

  function startNewChat() {
    setSessionId(null);
    setMessages([]);
  }

  async function sendMessage(e: React.FormEvent) {
    e.preventDefault();
    if (!input.trim() || !selectedArquetipo || loading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      rol: 'usuario',
      contenido: input.trim(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await interaccionApi.chat(
        selectedArquetipo,
        userMessage.contenido,
        sessionId || undefined
      );

      if (!sessionId) {
        setSessionId(response.session_id);
      }

      const botMessage: Message = {
        id: Date.now().toString() + '-bot',
        rol: 'sintetico',
        contenido: response.respuesta,
        fricciones: response.fricciones,
        emociones: response.emociones,
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage: Message = {
        id: Date.now().toString() + '-error',
        rol: 'sintetico',
        contenido: 'Error al enviar el mensaje. Intenta de nuevo.',
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  }

  const selectedArquetipoData = arquetipos.find(a => a.id === selectedArquetipo);

  return (
    <div className="h-[calc(100vh-7rem)] flex gap-6">
      {/* Sidebar */}
      <div className="w-80 flex flex-col bg-bg-secondary rounded-xl border border-border">
        <div className="p-4 border-b border-border">
          <h2 className="font-semibold text-text-primary mb-3">Selecciona un arquetipo</h2>
          <select
            value={selectedArquetipo}
            onChange={(e) => {
              setSelectedArquetipo(e.target.value);
              startNewChat();
            }}
            className="w-full px-3 py-2 rounded-lg border border-border bg-bg-primary text-text-primary focus:outline-none focus:ring-2 focus:ring-primary/50"
          >
            <option value="">Seleccionar...</option>
            {arquetipos.map(a => (
              <option key={a.id} value={a.id}>{a.nombre}</option>
            ))}
          </select>
        </div>

        {selectedArquetipoData && (
          <div className="p-4 flex-1 overflow-auto">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 bg-primary/10 rounded-lg">
                <User size={24} className="text-primary" />
              </div>
              <div>
                <h3 className="font-medium text-text-primary">{selectedArquetipoData.nombre}</h3>
                <p className="text-sm text-text-muted">
                  {selectedArquetipoData.edad && `${selectedArquetipoData.edad} anos`}
                  {selectedArquetipoData.edad && selectedArquetipoData.ocupacion && ' - '}
                  {selectedArquetipoData.ocupacion}
                </p>
              </div>
            </div>
            <p className="text-sm text-text-secondary mb-4">{selectedArquetipoData.descripcion}</p>

            <button
              onClick={startNewChat}
              className="w-full flex items-center justify-center gap-2 px-3 py-2 text-sm border border-border hover:bg-bg-tertiary rounded-lg transition-colors"
            >
              <Plus size={16} />
              Nueva conversacion
            </button>
          </div>
        )}
      </div>

      {/* Chat */}
      <div className="flex-1 flex flex-col bg-bg-secondary rounded-xl border border-border">
        {!selectedArquetipo ? (
          <div className="flex-1 flex items-center justify-center text-text-muted">
            Selecciona un arquetipo para comenzar
          </div>
        ) : (
          <>
            {/* Messages */}
            <div className="flex-1 overflow-auto p-4 space-y-4">
              {messages.length === 0 && (
                <div className="text-center text-text-muted py-8">
                  Inicia la conversacion con {selectedArquetipoData?.nombre}
                </div>
              )}
              {messages.map(message => (
                <div
                  key={message.id}
                  className={`flex gap-3 ${message.rol === 'usuario' ? 'flex-row-reverse' : ''}`}
                >
                  <div className={`p-2 rounded-lg ${
                    message.rol === 'usuario' ? 'bg-primary' : 'bg-accent'
                  }`}>
                    {message.rol === 'usuario' ? (
                      <User size={20} className="text-white" />
                    ) : (
                      <Bot size={20} className="text-white" />
                    )}
                  </div>
                  <div className={`max-w-[70%] ${message.rol === 'usuario' ? 'text-right' : ''}`}>
                    <div className={`px-4 py-3 rounded-2xl ${
                      message.rol === 'usuario'
                        ? 'bg-primary text-white rounded-tr-sm'
                        : 'bg-bg-tertiary text-text-primary rounded-tl-sm'
                    }`}>
                      {message.contenido}
                    </div>
                    {message.fricciones && message.fricciones.length > 0 && (
                      <div className="mt-2 text-xs text-warning">
                        Fricciones: {message.fricciones.join(', ')}
                      </div>
                    )}
                  </div>
                </div>
              ))}
              {loading && (
                <div className="flex gap-3">
                  <div className="p-2 rounded-lg bg-accent">
                    <Bot size={20} className="text-white" />
                  </div>
                  <div className="px-4 py-3 rounded-2xl rounded-tl-sm bg-bg-tertiary">
                    <Loader2 size={20} className="animate-spin text-text-muted" />
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <form onSubmit={sendMessage} className="p-4 border-t border-border">
              <div className="flex gap-3">
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Escribe tu mensaje..."
                  className="flex-1 px-4 py-2.5 rounded-lg border border-border bg-bg-primary text-text-primary focus:outline-none focus:ring-2 focus:ring-primary/50"
                  disabled={loading}
                />
                <button
                  type="submit"
                  disabled={!input.trim() || loading}
                  className="px-4 py-2.5 bg-primary hover:bg-primary-dark text-white rounded-lg transition-colors disabled:opacity-50"
                >
                  <Send size={20} />
                </button>
              </div>
            </form>
          </>
        )}
      </div>
    </div>
  );
}
