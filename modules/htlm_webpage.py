import streamlit as st
import streamlit.components.v1 as components

def display_bpmn_xml(bpmn_xml, vizi_file, is_mobile=False, screen_width=300):

    if is_mobile:
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>BPMN Modeler</title>
            <script src="https://unpkg.com/bpmn-js/dist/bpmn-modeler.development.js"></script>
            <style>
                html, body {{
                    height: 100%;
                    padding: 0;
                    margin: 0;
                    font-family: Arial, sans-serif;
                    display: flex;
                    flex-direction: column;
                    overflow: hidden;
                }}
                #button-container {{
                    padding: 10px;
                    background-color: #ffffff;
                    border-bottom: 1px solid #ddd;
                    display: flex;
                    justify-content: flex-start;
                    gap: 10px;
                }}
                #save-button, #download-button, #download-vizi-button {{
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    text-align: center;
                    text-decoration: none;
                    display: inline-block;
                    font-size: 16px;
                    margin: 4px 2px;
                    cursor: pointer;
                    border-radius: 8px;
                }}
                #download-button {{
                    background-color: #008CBA;
                }}
                #download-vizi-button {{
                    background-color: #FFA500;
                }}
                #canvas-container {{
                    flex: 1;
                    position: relative;
                    background-color: #FBFBFB;
                    overflow: hidden; /* Prevent scrolling */
                    display: flex;
                    justify-content: center;
                    align-items: center;
                }}
                #canvas {{
                    height: 100%;
                    width: 100%;
                    position: relative;
                }}
            </style>
        </head>
        <body>
            <div id="button-container">
                <button id="save-button">Save as BPMN</button>
                <button id="download-button">Save as XML</button>
                <button id="download-vizi-button">Save as Vizi</button>
            </div>
            <div id="canvas-container">
                <div id="canvas"></div>
            </div>
            <script>
                var bpmnModeler = new BpmnJS({{
                    container: '#canvas'
                }});

                async function openDiagram(bpmnXML) {{
                    try {{
                        await bpmnModeler.importXML(bpmnXML);
                        bpmnModeler.get('canvas').zoom('fit-viewport', 'auto');
                        bpmnModeler.get('canvas').zoom(0.2); // Adjust this value for zooming out
                    }} catch (err) {{
                        console.error('Error rendering BPMN diagram', err);
                    }}
                }}

                async function saveDiagram() {{
                    try {{
                        const result = await bpmnModeler.saveXML({{ format: true }});
                        const xml = result.xml;
                        const blob = new Blob([xml], {{ type: 'text/xml' }});
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = 'diagram.bpmn';
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                    }} catch (err) {{
                        console.error('Error saving BPMN diagram', err);
                    }}
                }}

                async function downloadXML() {{
                    try {{
                        const result = await bpmnModeler.saveXML({{ format: true }});
                        const xml = result.xml;
                        const blob = new Blob([xml], {{ type: 'text/xml' }});
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = 'diagram.xml';
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                    }} catch (err) {{
                        console.error('Error downloading XML', err);
                    }}
                }}

                async function downloadVizi() {{
                    try {{
                        const blob = new Blob([`{vizi_file}`], {{ type: 'text/plain' }});
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = 'vizi_file.pmw';
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                    }} catch (err) {{
                        console.error('Error downloading Vizi file', err);
                    }}
                }}

                document.getElementById('save-button').addEventListener('click', saveDiagram);
                document.getElementById('download-button').addEventListener('click', downloadXML);
                document.getElementById('download-vizi-button').addEventListener('click', downloadVizi);

                // Ensure the canvas is focused to capture keyboard events
                document.getElementById('canvas').focus();

                // Add event listeners for keyboard shortcuts
                document.addEventListener('keydown', function(event) {{
                    if (event.ctrlKey && event.key === 'z') {{
                        bpmnModeler.get('commandStack').undo();
                    }} else if (event.key === 'Delete' || event.key === 'Backspace') {{
                        bpmnModeler.get('selection').get().forEach(function(element) {{
                            bpmnModeler.get('modeling').removeElements([element]);
                        }});
                    }}
                }});

                openDiagram(`{bpmn_xml}`);
            </script>
        </body>
        </html>
        """
        
        components.html(html_template, height=screen_width, width=screen_width)
   
    else:
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>BPMN Modeler</title>
            <link rel="stylesheet" href="https://unpkg.com/bpmn-js/dist/assets/diagram-js.css">
            <link rel="stylesheet" href="https://unpkg.com/bpmn-js/dist/assets/bpmn-font/css/bpmn-embedded.css">
            <script src="https://unpkg.com/bpmn-js/dist/bpmn-modeler.development.js"></script>
            <style>
                html, body {{
                    height: 100%;
                    padding: 0;
                    margin: 0;
                    font-family: Arial, sans-serif;
                    display: flex;
                    flex-direction: column;
                    overflow: hidden;
                }}
                #button-container {{
                    padding: 10px;
                    background-color: #ffffff;
                    border-bottom: 1px solid #ddd;
                    display: flex;
                    justify-content: flex-start;
                    gap: 10px;
                }}
                #save-button, #download-button, #download-vizi-button {{
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    text-align: center;
                    text-decoration: none;
                    display: inline-block;
                    font-size: 16px;
                    margin: 4px 2px;
                    cursor: pointer;
                    border-radius: 8px;
                }}
                #download-button {{
                    background-color: #008CBA;
                }}
                #download-vizi-button {{
                    background-color: #FFA500;
                }}
                #canvas-container {{
                    flex: 1;
                    position: relative;
                    background-color: #FBFBFB;
                    overflow: hidden;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                }}
                #canvas {{
                    height: 100%;
                    width: 100%;
                    position: relative;
                }}
            </style>
        </head>
        <body>
            <div id="button-container">
                <button id="save-button">Save as BPMN</button>
                <button id="download-button">Save as XML</button>
                <button id="download-vizi-button">Save as Vizi</button>
            </div>
            <div id="canvas-container">
                <div id="canvas"></div>
            </div>
            <script>
                var bpmnModeler = new BpmnJS({{
                    container: '#canvas'
                }});

                async function openDiagram(bpmnXML) {{
                    try {{
                        await bpmnModeler.importXML(bpmnXML);
                        bpmnModeler.get('canvas').zoom('fit-viewport', 'auto');
                        bpmnModeler.get('canvas').zoom(0.8);
                    }} catch (err) {{
                        console.error('Error rendering BPMN diagram', err);
                    }}
                }}

                async function saveDiagram() {{
                    try {{
                        const result = await bpmnModeler.saveXML({{ format: true }});
                        const xml = result.xml;
                        const blob = new Blob([xml], {{ type: 'text/xml' }});
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = 'diagram.bpmn';
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                    }} catch (err) {{
                        console.error('Error saving BPMN diagram', err);
                    }}
                }}

                async function downloadXML() {{
                    try {{
                        const result = await bpmnModeler.saveXML({{ format: true }});
                        const xml = result.xml;
                        const blob = new Blob([xml], {{ type: 'text/xml' }});
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = 'diagram.xml';
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                    }} catch (err) {{
                        console.error('Error downloading XML', err);
                    }}
                }}

                async function downloadVizi() {{
                    try {{
                        const blob = new Blob([`{vizi_file}`], {{ type: 'text/plain' }});
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = 'vizi_file.pmw';
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                    }} catch (err) {{
                        console.error('Error downloading Vizi file', err);
                    }}
                }}

                document.getElementById('save-button').addEventListener('click', saveDiagram);
                document.getElementById('download-button').addEventListener('click', downloadXML);
                document.getElementById('download-vizi-button').addEventListener('click', downloadVizi);

                document.getElementById('canvas').focus();

                document.addEventListener('keydown', function(event) {{
                    if (event.ctrlKey && event.key === 'z') {{
                        bpmnModeler.get('commandStack').undo();
                    }} else if (event.key === 'Delete' || event.key === 'Backspace') {{
                        bpmnModeler.get('selection').get().forEach(function(element) {{
                            bpmnModeler.get('modeling').removeElements([element]);
                        }});
                    }}
                }});

                openDiagram(`{bpmn_xml}`);
            </script>
        </body>
        </html>
        """
        
        components.html(html_template, height=1000, width=int(9/10*screen_width))

