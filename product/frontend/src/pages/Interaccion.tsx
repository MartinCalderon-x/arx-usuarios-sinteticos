import { useEffect, useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { Send, User, Bot, Loader2, Plus, Users, Sparkles } from 'lucide-react';
import { arquetiposApi, interaccionApi, type Arquetipo, type ArquetipoTemplate } from '../lib/api';

interface Message {
  id: string;
  rol: 'usuario' | 'sintetico';
  contenido: string;
  fricciones?: string[];
  emociones?: Record<string, unknown>;
}

export function Interaccion() {
  const navigate = useNavigate();
  const [arquetipos, setArquetipos] = useState<Arquetipo[]>([]);
  const [templates, setTemplates] = useState<ArquetipoTemplate[]>([]);
  const [selectedArquetipo, setSelectedArquetipo] = useState<string>('');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [loadingArquetipos, setLoadingArquetipos] = useState(true);
  const [creatingFromTemplate, setCreatingFromTemplate] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    loadArquetipos();
    loadTemplates();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  async function loadArquetipos() {
    try {
      setLoadingArquetipos(true);
      const { arquetipos } = await arquetiposApi.list();
      setArquetipos(arquetipos);
    } catch (error) {
      console.error('Error loading arquetipos:', error);
    } finally {
      setLoadingArquetipos(false);
    }
  }

  async function loadTemplates() {
    try {
      const { templates } = await arquetiposApi.templates();
      setTemplates(templates.slice(0, 4)); // Show only 4 quick templates
    } catch (error) {
      console.error('Error loading templates:', error);
    }
  }

  async function createFromTemplate(template: ArquetipoTemplate) {
    try {
      setCreatingFromTemplate(true);
      const newArquetipo = await arquetiposApi.create({
        nombre: template.nombre,
        descripcion: template.descripcion,
        edad: template.edad,
        ocupacion: template.ocupacion,
        nivel_digital: template.nivel_digital,
        industria: template.industria,
        comportamiento: template.comportamiento,
        frustraciones: template.frustraciones,
        objetivos: template.objetivos,
      });
      setArquetipos(prev => [...prev, newArquetipo]);
      setSelectedArquetipo(newArquetipo.id);
    } catch (error) {
      console.error('Error creating archetype from template:', error);
    } finally {
      setCreatingFromTemplate(false);
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

          {loadingArquetipos ? (
            <div className="flex items-center justify-center py-4">
              <div className="w-6 h-6 border-2 border-primary/30 border-t-primary rounded-full animate-spin" />
            </div>
          ) : arquetipos.length > 0 ? (
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
          ) : (
            <div className="text-center py-2">
              <p className="text-sm text-text-muted mb-3">
                No tienes arquetipos creados
              </p>
              <button
                onClick={() => navigate('/arquetipos/nuevo')}
                className="w-full flex items-center justify-center gap-2 px-3 py-2 text-sm bg-primary hover:bg-primary-dark text-white rounded-lg transition-colors"
              >
                <Plus size={16} />
                Crear arquetipo
              </button>
            </div>
          )}
        </div>

        {/* Empty state with quick templates */}
        {!loadingArquetipos && arquetipos.length === 0 && templates.length > 0 && (
          <div className="p-4 flex-1 overflow-auto">
            <div className="flex items-center gap-2 mb-3">
              <Sparkles size={16} className="text-accent" />
              <h3 className="text-sm font-medium text-text-secondary">Creacion rapida</h3>
            </div>
            <p className="text-xs text-text-muted mb-3">
              Crea un arquetipo desde una plantilla para comenzar:
            </p>
            <div className="space-y-2">
              {templates.map(template => (
                <button
                  key={template.id}
                  onClick={() => createFromTemplate(template)}
                  disabled={creatingFromTemplate}
                  className="w-full text-left p-3 rounded-lg border border-border hover:bg-bg-tertiary transition-colors disabled:opacity-50"
                >
                  <div className="flex items-center gap-2">
                    <Users size={14} className="text-primary" />
                    <span className="text-sm font-medium text-text-primary">{template.nombre}</span>
                  </div>
                  <p className="text-xs text-text-muted mt-1 line-clamp-2">{template.descripcion}</p>
                </button>
              ))}
            </div>
            {creatingFromTemplate && (
              <div className="flex items-center justify-center gap-2 mt-3 text-sm text-text-muted">
                <Loader2 size={14} className="animate-spin" />
                Creando arquetipo...
              </div>
            )}
          </div>
        )}

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
          <div className="flex-1 flex flex-col items-center justify-center text-text-muted p-8">
            {loadingArquetipos ? (
              <div className="w-8 h-8 border-2 border-primary/30 border-t-primary rounded-full animate-spin" />
            ) : arquetipos.length === 0 ? (
              <div className="text-center">
                <Users size={48} className="mx-auto mb-4 text-text-muted/50" />
                <h3 className="text-lg font-medium text-text-primary mb-2">
                  Crea tu primer arquetipo
                </h3>
                <p className="text-sm text-text-muted mb-4 max-w-md">
                  Los arquetipos son usuarios sinteticos que simulan comportamientos reales.
                  Usa una plantilla rapida o crea uno personalizado.
                </p>
                <button
                  onClick={() => navigate('/arquetipos/nuevo')}
                  className="inline-flex items-center gap-2 px-4 py-2 bg-primary hover:bg-primary-dark text-white rounded-lg transition-colors"
                >
                  <Plus size={18} />
                  Crear arquetipo personalizado
                </button>
              </div>
            ) : (
              <>
                <User size={48} className="mb-4 text-text-muted/50" />
                <p>Selecciona un arquetipo para comenzar</p>
              </>
            )}
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
